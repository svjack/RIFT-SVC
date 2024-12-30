from huggingface_hub import snapshot_download


if __name__ == "__main__":
    snapshot_download(repo_id="Pur1zumu/RIFT-SVC-modules", local_dir='pretrained')
