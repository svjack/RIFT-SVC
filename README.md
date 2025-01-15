# RIFT-SVC: Rectified Flow Transformer for Singing Voice Conversion

<p align="center"><img src="./assets/logo.png" alt="RIFT-SVC" width="200"/></p>

 Implementation of **RIFT-SVC**, a singing voice conversion model based on Rectified Flow Transformer. We tested several architecture-wise and training-wise improvements over vanilla Diffusion Transformer.

## News
- 2025-01-14: V2.0 is released. The cfg-strength can now be used to control the balance between timbre and articulation.
- 2024-12-30: V1.0 is released.

---
## Environment Preparation

#### 1. Clone the repository
```bash
git clone https://github.com/Pur1zumu/RIFT-SVC.git
cd RIFT-SVC
```

#### 2. Create a new conda environment
```bash
conda create -n rift-svc python=3.11
conda activate rift-svc
```

#### 3. Install torch that supports your cuda version. See [PyTorch](https://pytorch.org/get-started/locally/) for more details.
E.g., for cuda 12.1, use:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Install other dependencies
```bash
pip install -r requirements.txt
```

## Models and Data Preparation

#### 5. Download pretrained models for feature extraction and vocoder.
```bash
python pretrained/download.py
```

#### 6. Download pretrained weights for fine-tuning.

| Model | Command |
| --- | --- |
| pretrain-v2-final_dit-768-12_300000steps-lr0.0003 | wget https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/pretrain-v2-final_dit-768-12_300000steps-lr0.0003.ckpt -O pretrained/pretrain-v2-final_dit-768-12_300000steps-lr0.0003.ckpt |


#### 7. Prepare data and extract features.
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
bash prepare.sh data/finetune 1
```
where `1` is the number of workers per device. You can adjust this value based on your GPU memory (like 4 for 3090).


## Training

#### 8. Start Finetuning

To monitor the training process, we use wandb. 
```bash
wandb login
```
You can find the training logs in the wandb dashboard. See [wandb](https://wandb.ai/) for more details.

Run the following command to start fine-tuning:
```bash
python train.py --config-name finetune model=dit-768-12 training.wandb_run_name=finetune_ckpt-v2_dit-768-12_30000steps-lr0.00005 training.learning_rate=5e-5 +model.lognorm=true training.max_steps=30000 training.weight_decay=0.01 training.batch_size_per_gpu=64 training.save_per_steps=1000 training.test_per_steps=1000 +model.pretrained_path=pretrained/pretrain-v2-final_dit-768-12_300000steps-lr0.0003.ckpt +model.whisper_drop_prob=0.2 training.eval_cfg_strength=2.0
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
- `--model.whisper_drop_prob`: The probability of dropping out the whisper embedding.
- `--training.eval_cfg_strength`: The strength of classifier-free guidance for evaluation.
- `--training.eval_sample_steps`: The number of inference steps to sample for evaluation.

Since we use hydra to manage the config, you can also override the config file by adding your own arguments.

The above settings are tested on one 3090 GPU, consuming ~20GB VRAM.

### Resume from checkpoint
If you expect to resume training from a checkpoint, you can add the following argument:
```bash
[other arguments] +training.save_weights_only=False
```

Then, you can resume training from a checkpoint later by adding the following argument:
```bash
[other arguments] +training.resume_from_checkpoint=/path/to/checkpoint.ckpt +training.wandb_resume_id=your_wandb_run_id
```

## Inference

#### 9. Inference
```bash
python infer.py --model ckpts/finetune_ckpt-v2_dit-768-12_30000steps-lr0.00005/model-step=24000.ckpt --input 0.wav --output 0_steps32_cfg0.wav --speaker speaker1 --infer-steps 32 --cfg-strength 2.0
```

**Arguments:**
Arguments:
- `--model`: The path to the fine-tuned model.
- `--input`: The path to the input audio file.
- `--output`: The path to the output audio file.
- `--speaker`: The speaker name.
- `--infer-steps`: The number of inference steps.
- `--cfg-strength`: The strength of classifier-free guidance. A lower value (such as 0) produces a timbre closer to the target speaker, but with relatively less clear articulation and lower noise resistance. A higher value (such as 5) produces clearer articulation and better noise resistance, but the style will be closer to the input source. We set a balance value of 2.0 by default.

For 768-12 model, the VRAM consumption is ~5GB.

---

## Key Technical Details

### V2
#### Major Gains
- Add whisper encoder[9] and use the hidden state from the second last layer to reduce timbre leakage, as explored in [10]
- Add classifier-free guidance training by randomly dropping out the whisper embedding
- Add classifier-free guidance inference by f(cvec, null) + cfg_strength * (f(cvec, speaker) - f(cvec, null)) in each sample step
    - Note that this is different from the original classifier-free guidance in Diffusion Transformer, which is f(condition) + cfg_strength * (f(condition) - f(null))
- Add post-LayerNorm at appropriate positions in the input embedding module [11]

#### Minor Gains
- Add gating module for contentvec embedding and whisper embedding

### V1
#### Major Gains

- NSF-HIFIGAN[1] instead of BigVGAN[2]
- Add depth-wise conv in MLP, kind of like ConvNext block in 1d[3]
- Spectral parameterization (in terms of init and learning rate)[4]
- Scheduler-free optimizer (have not compared with vanilla AdamW, but no need to tune the scheduler anyway)[5]

#### Minor Gains

- LogNorm noise scheduler[6]
- Zero init for output blocks
- Classifier-free guidance training by randomly dropping out the speaker embedding (no quantitative test yet, but theoretically it should be better. But it not works for single speaker training.) [Removed in V2]
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

[9] Whisper encoder: https://huggingface.co/openai/whisper-large-v3

[10] Zhang, Li, et al. "Whisper-SV: Adapting Whisper for low-data-resource speaker verification." Speech Communication 163 (2024): 103103.

[11] Li, Pengxiang, Lu Yin, and Shiwei Liu. "Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN." arXiv preprint arXiv:2412.13795 (2024).
