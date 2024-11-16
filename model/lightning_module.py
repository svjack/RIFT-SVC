from pytorch_lightning import LightningModule
import wandb
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import gc
import os

from model.utils import exists, default
from model.modules import get_mel_spectrogram
import model.BigVGAN.bigvgan as bigvgan
from model.metrics import mcd, si_snr, psnr, snr


class RIFTSVCLightningModule(LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        eval_sample_steps: int = 32,
        eval_cfg_strength: float = 2.0,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.eval_sample_steps = eval_sample_steps
        self.eval_cfg_strength = eval_cfg_strength
        # Initialize vocoder
        self.vocoder = None
        
        # Save hyperparameters
        #self.save_hyperparameters(ignore=['model'])

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        mel_spec = batch['mel_spec']
        spk_id = batch['spk_id']
        f0 = batch['f0']
        rms = batch['rms']
        cvec = batch['cvec']
        frame_lens = batch['frame_lens']

        loss, pred = self.model(
            mel_spec,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            lens=frame_lens,
        )

        self.log('train/loss', loss, prog_bar=True, logger=True)
        return loss
    
    def on_validation_start(self):
        if not self.trainer.is_global_zero:
            return

        if self.vocoder is None:
            self.vocoder = bigvgan.BigVGAN.from_pretrained(
                'pretrained/bigvgan_v2_44khz_128band_256x', 
                use_cuda_kernel=False
            ).eval().float().to(self.device)
            self.vocoder.remove_weight_norm()
        else:
            self.vocoder = self.vocoder.to(self.device)
        
        self.mcd = []
        self.si_snr = []
        self.psnr = []
        self.snr = []
        self.mse = []

    def on_validation_end(self):
        if not self.trainer.is_global_zero:
            return

        if hasattr(self, 'vocoder'):
            self.vocoder = self.vocoder.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        
        metrics = {
            'val/mcd': np.mean(self.mcd),
            'val/si_snr': np.mean(self.si_snr),
            'val/psnr': np.mean(self.psnr),
            'val/snr': np.mean(self.snr),
            'val/mse': np.mean(self.mse)
        }
        self.logger.experiment.log(metrics, step=self.global_step)


    def validation_step(self, batch, batch_idx):
        if not self.trainer.is_global_zero:
            return
        
        global_step = self.trainer.global_step
        save_every_n_steps = self.trainer.callbacks[0].every_n_train_steps

        spk_id = batch['spk_id']
        mel_gt = batch['mel_spec']
        rms = batch['rms']
        f0 = batch['f0']
        cvec = batch['cvec']
        frame_lens = batch['frame_lens']

        mel_gen, _ = self.model.sample(
            src_mel=mel_gt,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            frame_lens=frame_lens,
            steps=self.eval_sample_steps,
            cfg_strength=self.eval_cfg_strength,
        )
        mel_gen = mel_gen.float()
        mel_gt = mel_gt.float()

        for i in range(mel_gen.shape[0]):
            wav_gen = self.vocoder(mel_gen[i:i+1, :frame_lens[i], :].transpose(1, 2)).squeeze(0)
            wav_gt = self.vocoder(mel_gt[i:i+1, :frame_lens[i], :].transpose(1, 2)).squeeze(0)

            sample_idx = batch_idx * mel_gen.shape[0] + i

            mel_gen_i = get_mel_spectrogram(wav_gen).transpose(1, 2)
            mel_gt_i = get_mel_spectrogram(wav_gt).transpose(1, 2)

            mel_min, mel_max = self.model.mel_min, self.model.mel_max
            mel_gen_i = torch.clip(mel_gen_i, min=mel_min, max=mel_max)
            mel_gt_i = torch.clip(mel_gt_i, min=mel_min, max=mel_max)

            self.mcd.append(mcd(mel_gen_i, mel_gt_i).cpu().item())
            self.si_snr.append(si_snr(mel_gen_i, mel_gt_i).cpu().item())
            self.psnr.append(psnr(mel_gen_i, mel_gt_i).cpu().item())
            self.snr.append(snr(mel_gen_i, mel_gt_i).cpu().item())
            self.mse.append(F.mse_loss(mel_gen_i, mel_gt_i).cpu().item())

            os.makedirs('.cache', exist_ok=True)
            if global_step % save_every_n_steps == 0:
                torchaudio.save(f".cache/{sample_idx}_gen.wav", wav_gen.cpu().to(torch.float32), 44100)
                self.logger.experiment.log({
                    f"val-audio/{sample_idx}_gen": wandb.Audio(f".cache/{sample_idx}_gen.wav", sample_rate=44100),
                }, step=self.global_step)
            
            if global_step == 0:
                torchaudio.save(f".cache/{sample_idx}_gt.wav", wav_gt.cpu().to(torch.float32), 44100)
                self.logger.experiment.log({
                    f"val-audio/{sample_idx}_gt": wandb.Audio(f".cache/{sample_idx}_gt.wav", sample_rate=44100)
                }, step=self.global_step)

            if global_step % save_every_n_steps == 0:
                # Compute global min and max for consistent scaling across all plots
                data_gt = mel_gt_i.squeeze().T.cpu().numpy()
                data_gen = mel_gen_i.squeeze().T.cpu().numpy()
                data_diff = data_gen - data_gt
                vmin = min(data_gt.min(), data_gen.min(), data_diff.min())
                vmax = max(data_gt.max(), data_gen.max(), data_diff.max())
                
                # Create figure with space for colorbar
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), sharex=True, gridspec_kw={'hspace': 0})
                
                # Plot all spectrograms with the same scale
                im1 = ax1.imshow(data_gt, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
                ax1.set_ylabel('GT', fontsize=14)
                ax1.set_xticks([])
                
                im2 = ax2.imshow(data_gen, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
                ax2.set_ylabel('Gen', fontsize=14)
                ax2.set_xticks([])
                
                im3 = ax3.imshow(data_diff, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
                ax3.set_ylabel('Diff', fontsize=14)
                
                # Add single shared colorbar
                fig.colorbar(im1, ax=[ax1, ax2, ax3], location='right')
                
                plt.savefig(f".cache/{sample_idx}_mel.jpg", quality=85, bbox_inches='tight')
                plt.close()

                self.logger.experiment.log({
                    f"val-mel/{sample_idx}_mel": wandb.Image(f".cache/{sample_idx}_mel.jpg")
                }, step=self.global_step)
