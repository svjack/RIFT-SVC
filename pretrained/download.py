from huggingface_hub import hf_hub_download
import os

def download_pretrained_models():
    # Repository and directories to download
    repo_id = "Pur1zumu/RIFT-SVC-pretrained"
    directories = [
        "content-vec-best",
        "nsf_hifigan_44.1k_hop512_128bin_2024.02",
        "rmvpe"
    ]
    
    # Create base directory if it doesn't exist
    os.makedirs("pretrained", exist_ok=True)
    
    for dir_name in directories:
        try:
            print(f"Downloading {dir_name}...")
            
            # Download all files from the directory
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{dir_name}/*",
                local_dir=".",
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"Successfully downloaded {dir_name}")
            
        except Exception as e:
            print(f"Error downloading {dir_name}: {str(e)}")

if __name__ == "__main__":
    print("Starting download of pretrained models...")
    download_pretrained_models()
    print("Download process completed!")