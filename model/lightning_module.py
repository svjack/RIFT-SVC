import gc
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import wandb
from PIL import Image
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm

import model.BigVGAN.bigvgan as bigvgan
from model.metrics import mcd, psnr, si_snr, snr
from model.modules import get_mel_spectrogram
from model.utils import draw_mel_specs, l2_grad_norm


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
        self.optimizer.eval()
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
        self.optimizer.train()
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
        
        global_step = self.global_step
        log_media_every_n_steps = self.log_media_every_n_steps

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
            if global_step % log_media_every_n_steps == 0:
                torchaudio.save(f".cache/{sample_idx}_gen.wav", wav_gen.cpu().to(torch.float32), 44100)
                self.logger.experiment.log({
                    f"val-audio/{sample_idx}_gen": wandb.Audio(f".cache/{sample_idx}_gen.wav", sample_rate=44100),
                }, step=self.global_step)
            
            if global_step == 0:
                torchaudio.save(f".cache/{sample_idx}_gt.wav", wav_gt.cpu().to(torch.float32), 44100)
                self.logger.experiment.log({
                    f"val-audio/{sample_idx}_gt": wandb.Audio(f".cache/{sample_idx}_gt.wav", sample_rate=44100)
                }, step=self.global_step)

            if global_step % log_media_every_n_steps == 0:
                # Compute global min and max for consistent scaling across all plots
                data_gt = mel_gt_i.squeeze().T.cpu().numpy()
                data_gen = mel_gen_i.squeeze().T.cpu().numpy()
                data_abs_diff = data_gen - data_gt

                cache_path = f".cache/{sample_idx}_mel.jpg"
                draw_mel_specs(data_gt, data_gen, data_abs_diff, cache_path)

                self.logger.experiment.log({
                    f"val-mel/{sample_idx}_mel": wandb.Image(cache_path)
                }, step=self.global_step)

    def on_before_optimizer_step(self, optimizer):
        # Calculate gradient norm
        norm = l2_grad_norm(self.model)

        self.log('train/grad_norm', norm, prog_bar=True, logger=True)

    @property
    def global_step(self):
        return self.trainer.global_step

    @property
    def log_media_every_n_steps(self):
        if self.save_every_n_steps is None:
            return self.trainer.val_check_interval
        return self.save_every_n_steps
    
    @property
    def save_every_n_steps(self):
        for callback in self.trainer.callbacks:
            if hasattr(callback, '_every_n_train_steps'):
                return callback._every_n_train_steps
        return None