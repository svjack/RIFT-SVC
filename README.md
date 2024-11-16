# RIFT-SVC

## Environment

1. Create a new conda environment
```bash
conda create -n rift-svc python=3.11
conda activate rift-svc
```

2. Install torch that supports your cuda version. See [PyTorch](https://pytorch.org/get-started/locally/) for more details.
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Install other dependencies
```bash
pip install -r requirements.txt
```
