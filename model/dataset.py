import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def interpolate_tensor(tensor, new_size):
    # Add two dummy dimensions to make it [1, 1, n, d]
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    # Interpolate
    interpolated = F.interpolate(tensor, size=(new_size, tensor.shape[-1]), mode='nearest')
    
    # Remove the dummy dimensions
    interpolated = interpolated.squeeze(0).squeeze(0)
    
    return interpolated


class SVCDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        meta_info_path: str,
        split = "train",
        max_frame_len = 1024,
    ):
        self.data_dir = data_dir
        self.max_frame_len = max_frame_len

        with open(meta_info_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        speakers = meta["speakers"]
        self.num_speakers = len(speakers)
        self.spk2idx = {spk: idx for idx, spk in enumerate(speakers)}
        self.split = split
        self.samples = meta[f"{split}_audios"]

    def get_frame_len(self, index):
        return self.samples[index]['frame_len']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):

        sample = self.samples[index]
        spk = sample['speaker']
        path = os.path.join(self.data_dir, spk, sample['file_name'])
        spk_id = torch.LongTensor([self.spk2idx[spk]]) # [1]
        # mel_spec = torch.load(path + ".mel.pt", weights_only=True).squeeze(0).T # [T, C]
        # rms = torch.load(path + ".rms.pt", weights_only=True).squeeze(0) # [T]
        # f0 = torch.load(path + ".f0.pt", weights_only=True).squeeze(0) # [T]
        # cvec = torch.load(path + ".cvec.pt", weights_only=True).squeeze(0) # [N, D]
        combined = torch.load(path + ".combined.pt", weights_only=True, map_location=torch.device('cpu'), mmap=True)
        mel_spec = combined.get('mel').squeeze(0).T
        rms = combined.get('rms').squeeze(0)
        f0 = combined.get('f0').squeeze(0)
        cvec = combined.get('cvec').squeeze(0)

        cvec = interpolate_tensor(cvec, mel_spec.shape[0]) # [T, D]
        frame_len = mel_spec.shape[0]

        if frame_len > self.max_frame_len:
            if self.split == "train": 
                # Keep trying until we find a good segment or hit max attempts
                max_attempts = 10
                attempt = 0
                while attempt < max_attempts:
                    start = random.randint(0, frame_len - self.max_frame_len)
                    end = start + self.max_frame_len
                    f0_segment = f0[start:end]
                    # Check if more than 90% of f0 values are 0
                    zero_ratio = (f0_segment == 0).float().mean().item()
                    if zero_ratio < 0.9:  # Found a good segment
                        break
                    attempt += 1
            else:
                start = 0
            end = start + self.max_frame_len
            mel_spec = mel_spec[start:end]
            rms = rms[start:end]
            f0 = f0[start:end]
            cvec = cvec[start:end]
            frame_len = self.max_frame_len

        result = dict(
            spk_id = spk_id,
            mel_spec = mel_spec,
            rms = rms,
            f0 = f0,
            cvec = cvec,
            frame_len = frame_len
        )

        return result
    

# load dataset

def load_svc_dataset(data_dir: str, meta_info_path: str, split = "train", max_frame_len = 1024) -> SVCDataset:
    return SVCDataset(data_dir, meta_info_path, split, max_frame_len)

# collation

# def collate_fn(batch):
#     # Separate different features from the batch
#     spk_ids = [item['spk_id'] for item in batch]

#     # Get the maximum length in the batch
#     frame_lens = [item['frame_len'] for item in batch]

#     max_frame_len = max(frame_lens)
#     b = len(batch)
#     # Pad sequences to max length
#     mel_specs_padded = torch.zeros(b, max_frame_len, batch[0]['mel_spec'].shape[-1])
#     rmss_padded = torch.zeros(b, max_frame_len)
#     f0s_padded = torch.zeros(b, max_frame_len)
#     cvecs_padded = torch.zeros(b, max_frame_len, batch[0]['cvec'].shape[-1])

#     for i, item in enumerate(batch):
#         length = item['frame_len']
#         mel_specs_padded[i, :length] = item['mel_spec']
#         rmss_padded[i, :length] = item['rms']
#         f0s_padded[i, :length] = item['f0']
#         cvecs_padded[i, :length] = item['cvec']

#     # Stack speaker IDs
#     spk_ids = torch.cat(spk_ids)
#     frame_lens = torch.tensor(frame_lens)

#     return {
#         'spk_id': spk_ids,
#         'mel_spec': mel_specs_padded,
#         'rms': rmss_padded,
#         'f0': f0s_padded,
#         'cvec': cvecs_padded,
#         'frame_lens': frame_lens
#     }

def collate_fn(batch):
    # Separate different features from the batch
    spk_ids = [item['spk_id'] for item in batch]
    mel_specs = [item['mel_spec'] for item in batch]
    rmss = [item['rms'] for item in batch]
    f0s = [item['f0'] for item in batch]
    cvecs = [item['cvec'] for item in batch]

    # Get the maximum length in the batch
    frame_lens = [item['frame_len'] for item in batch]

    # Pad sequences to max length
    mel_specs_padded = pad_sequence(mel_specs, batch_first=True)
    rmss_padded = pad_sequence(rmss, batch_first=True)
    f0s_padded = pad_sequence(f0s, batch_first=True)
    cvecs_padded = pad_sequence(cvecs, batch_first=True)

    # Stack speaker IDs
    spk_ids = torch.cat(spk_ids)
    frame_lens = torch.tensor(frame_lens)

    return {
        'spk_id': spk_ids,
        'mel_spec': mel_specs_padded,
        'rms': rmss_padded,
        'f0': f0s_padded,
        'cvec': cvecs_padded,
        'frame_lens': frame_lens
    }