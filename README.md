# RIFT-SVC: Rectified Flow Transformer for Singing Voice Conversion

<p align="center"><img src="./assets/logo.png" alt="RIFT-SVC" width="150"/></p>

 Implementation of **RIFT-SVC**, a singing voice conversion model based on Rectified Flow Transformer. We tested several architecture-wise and training-wise improvements over vanilla Diffusion Transformer.


---
## Environment

#### 1. Create a new conda environment
```bash
conda create -n rift-svc python=3.11
conda activate rift-svc
```

#### 2. Install torch that supports your cuda version. See [PyTorch](https://pytorch.org/get-started/locally/) for more details.
E.g., for cuda 12.1, use:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Install other dependencies
```bash
pip install -r requirements.txt
```

## Models and Data Preparation

#### 4. Download pretrained models for feature extraction and vocoder.
```bash
python pretrained/download.py
```

#### 5. Download pretrained weights for fine-tuning (Optional).

| Model | Command |
| --- | --- |
| pretrain-v3-final_dit-768-12_300000steps-lr0.0003 | wget https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/pretrain-v3-final_dit-768-12_300000steps-lr0.0003.ckpt -O pretrained/pretrain-v3-final_dit-768-12_300000steps-lr0.0003.ckpt |


#### 6. Prepare data and extract features.
You should structure your data like this:
```
data/
    finetune/
        speaker1/
            audio1.wav
            audio2.wav
            ...
        speaker2/
            audio1.wav
            audio2.wav
            ...
```
`data/finetune` is the default data directory for fine-tuning, but you can change it to your own data directory. Just make sure to adjust the corresponding arguments in other commands and scripts.

The audio files are expected to be resampled to 44100 Hz and loudness normalized to -18 LUFS by default.
You can use `scripts/resample_normalize_audios.py` to do this, for example:
```bash
python scripts/resample_normalize_audios.py --src data/finetune
```
or do it manually for the settings you want.

To extract features, run:
```bash
bash prepare.sh
```
You can adjust the DATA_DIR in `prepare.sh` to your own data directory. We use `data/finetune` by default.

## Training

#### 7. Start Finetuning
If one speaker is used, an example command is:
```bash
python train.py --config-name finetune model=dit-768-12 training.wandb_run_name=finetune_ckpt-v3_dit-768-12_30000steps-lr0.00005 training.learning_rate=5e-5 +model.lognorm=true training.max_steps=30000 training.weight_decay=0.01 training.batch_size_per_gpu=64 training.save_per_steps=1000 training.test_per_steps=1000 +model.pretrained_path=pretrained/pretrain-v3-final_dit-768-12_300000steps-lr0.0003.ckpt +model.spk_drop_prob=0.0 training.eval_cfg_strength=0.0
```

If multiple speakers are used:
```bash
python train.py --config-name finetune model=dit-768-12 training.wandb_run_name=finetune_ckpt-v3_dit-768-12_30000steps-lr0.00005 training.learning_rate=5e-5 +model.lognorm=true training.max_steps=30000 training.weight_decay=0.01 training.batch_size_per_gpu=64 training.save_per_steps=1000 training.test_per_steps=1000 +model.pretrained_path=pretrained/pretrain-v3-final_dit-768-12_300000steps-lr0.0003.ckpt +model.spk_drop_prob=0.2 training.eval_cfg_strength=2.0
```

**Arguments:**
- `--config-name finetune`: The name of the config file in `config/finetune.yaml`.
- `--model`: The model architecture to use for fine-tuning. See `config/model/` for more details. The model should align with the pretrained model.
- `--training.wandb_run_name`: The name of the wandb run.
- `--training.learning_rate`: The learning rate for fine-tuning.
- `--model.lognorm`: Whether to use lognorm noise scheduler.
- `--training.max_steps`: The maximum number of steps for fine-tuning.
- `--training.weight_decay`: The weight decay for fine-tuning.
- `--training.batch_size_per_gpu`: The batch size per GPU for fine-tuning. Adjust this value based on your GPU memory.
- `--training.save_per_steps`: The frequency of saving checkpoints.
- `--training.test_per_steps`: The frequency of testing.
- `--model.pretrained_path`: The path to the pretrained model.
- `--model.spk_drop_prob`: The probability of dropping out the speaker embedding. (Only used for multiple speakers fine-tuning)
- `--training.eval_cfg_strength`: The strength of classifier-free guidance for evaluation. (Only used for multiple speakers fine-tuning)
- `--training.eval_sample_steps`: The number of inference steps to sample for evaluation.

Since we use hydra to manage the config, you can also override the config file by adding your own arguments.

The above settings are tested on one 3090 GPU, consuming ~20GB VRAM.


To monitor the training process, we use wandb. You can find the training logs in the wandb dashboard. See [wandb](https://wandb.ai/) for more details.


## Inference

#### 8. Inference
```bash
python infer.py --model ckpts/finetune_ckpt-v3_dit-768-12_30000steps-lr0.00005/model-step=24000.ckpt --input 0.wav --output 0_steps32_cfg0.wav --speaker speaker1 --infer-steps 32 --cfg-strength 0.0
```

**Arguments:**
Arguments:
- `--model`: The path to the fine-tuned model.
- `--input`: The path to the input audio file.
- `--output`: The path to the output audio file.
- `--speaker`: The speaker name.
- `--infer-steps`: The number of inference steps.
- `--cfg-strength`: The strength of classifier-free guidance. If the model is trained with single speaker, you should set this to 0.0.

For 768-12 model, the VRAM consumption is ~4GB.

---

## Key Technical Details

#### Major Gains

- NSF-HIFIGAN[1] instead of BigVGAN[2]
- Add depth-wise conv in MLP, kind of like ConvNext block in 1d[3]
- Spectral parameterization (in terms of init and learning rate)[4]
- Scheduler-free optimizer (have not compared with vanilla AdamW, but no need to tune the scheduler anyway)[5]

#### Minor Gains

- LogNorm noise scheduler[6]
- Zero init for output blocks
- Classifier-free guidance training by randomly dropping out the speaker embedding (no quantitative test yet, but theoretically it should be better. But it not works for single speaker training.)
- Add an MLP for input condition embedding

#### No Gains

- UNet-like skip connections[7]
- MuP (I think it should be better, but I failed to pass the coordcheck. Now using Spectral parameterization instead)[8]

## References

[1] NSF-HIFIGAN: https://github.com/openvpi/SingingVocoders
[2] BigVGAN: https://github.com/NVIDIA/BigVGAN
[3] ConvNext: https://github.com/facebookresearch/ConvNeXt
[4] Spectral parameterization: Yang, G., Simon, J. B., & Bernstein, J. (2023). A spectral condition for feature learning. arXiv preprint arXiv:2310.17813.
[5] Scheduler-free optimizer: https://github.com/facebookresearch/schedule_free
[6] LogNorm noise scheduler: Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., ... & Rombach, R. (2024, March). Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning.
[7] UNet-like skip connections: Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., & Zhu, J. (2023). All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 22669-22679).
[8] MuP: Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., ... & Gao, J. (2022). Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466.