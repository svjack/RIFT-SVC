import os
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader

from rift_svc import DiT, RF
from rift_svc.dataset import SVCDataset, collate_fn
from rift_svc.lightning_module import RIFTSVCLightningModule
from rift_svc.utils import CustomProgressBar, load_state_dict

torch.set_float32_matmul_precision('high')


def get_optimizer(model, lr, betas, weight_decay, warmup_steps):
    from collections import defaultdict
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    specp_decay_params = defaultdict(list)
    specp_decay_lr = {}
    decay_params = []
    nodecay_params = []
    for n, p in param_dict.items():
        if p.dim() >= 2:
            if n.endswith('out.weight') or n.endswith('proj.weight'):
                fan_out, fan_in = p.shape[-2:]
                fan_ratio = fan_out / fan_in
                specp_decay_params[f"specp_decay_{fan_ratio:.2f}"].append(p)
                specp_decay_lr[f"specp_decay_{fan_ratio:.2f}"] = lr * fan_ratio
            else:
                decay_params.append(p)
        else:
            nodecay_params.append(p)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr},
        {'params': nodecay_params, 'weight_decay': 0.0, 'lr': lr}
    ] + [
        {'params': params, 'weight_decay': weight_decay, 'lr': specp_decay_lr[group_name]}
        for group_name, params in specp_decay_params.items()
    ]
    
    optimizer = AdamWScheduleFree(optim_groups, betas=betas, warmup_steps=warmup_steps)
    return optimizer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    train_dataset = SVCDataset(
        **cfg.dataset,
        split="train"
    )
    
    val_dataset = SVCDataset(
        **cfg.dataset,
        split="test"
    )

    transformer = DiT(
        **cfg.model,
        num_speaker=train_dataset.num_speakers,
    )

    rf = RF(
        transformer=transformer,
        time_schedule=cfg.training.time_schedule,
    )

    # Load pretrained weights if specified
    if cfg.training.get('pretrained_path', None) is not None:
        state_dict = torch.load(cfg.training.pretrained_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Load only model weights, allowing mismatched keys for speaker embeddings
        missing_keys, unexpected_keys = load_state_dict(rf, state_dict)
        print(f"Loaded pretrained model from {cfg.training.pretrained_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    warmup_steps = int(cfg.training.max_steps * cfg.training.warmup_ratio)
    optimizer = get_optimizer(
        rf, 
        cfg.training.learning_rate, 
        eval(cfg.training.betas), 
        cfg.training.weight_decay, 
        warmup_steps,
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['spk2idx'] = train_dataset.spk2idx
    model = RIFTSVCLightningModule(
        model=rf,
        optimizer=optimizer,
        cfg=cfg_dict
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('ckpts', cfg.training.wandb_run_name),
        filename='model-{step}',
        save_top_k=-1,
        save_last='link',
        every_n_train_steps=cfg.training.save_per_steps,
        save_weights_only=cfg.training.save_weights_only,
    )

    wandb_logger = WandbLogger(
        project=cfg.training.wandb_project,
        name=cfg.training.wandb_run_name,
        id=cfg.training.get('wandb_resume_id', None),
        resume='allow',
    )
    if wandb_logger.experiment.config:
        # Merge with existing config, giving priority to existing values
        wandb_logger.experiment.config.update(cfg_dict, allow_val_change=True)
    else:
        # If no existing config, set it directly
        wandb_logger.experiment.config.update(cfg_dict)

    callbacks = [checkpoint_callback, CustomProgressBar()]

    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator='gpu',
        devices='auto',
        strategy='auto',
        precision='bf16-mixed',
        accumulate_grad_batches=cfg.training.grad_accumulation_steps,
        callbacks=callbacks,
        logger=wandb_logger,
        val_check_interval=cfg.training.test_per_steps,
        check_val_every_n_epoch=None,
        gradient_clip_val=cfg.training.max_grad_norm,
        gradient_clip_algorithm='norm',
        log_every_n_steps=1,
    )

    if hasattr(optimizer, 'train'):
        optimizer.train()

    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
        ),
        ckpt_path=cfg.training.get('resume_from_checkpoint', None),
    )

if __name__ == "__main__":
    main()