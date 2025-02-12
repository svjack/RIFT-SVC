import json
import os
import random
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from rift_svc.utils import interpolate_tensor

pt_load = partial(torch.load, weights_only=True, map_location='cpu', mmap=True)


class SVCDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        meta_info_path: str,
        split = "train",
        max_frame_len = 256,
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

        mel = pt_load(path + ".mel.pt").squeeze(0).T
        rms = pt_load(path + ".rms.pt").squeeze(0)
        f0 = pt_load(path + ".f0.pt").squeeze(0)
        cvec = pt_load(path + ".cvec.pt").squeeze(0)
        whisper = pt_load(path + ".whisper.pt").squeeze(0)

        cvec = interpolate_tensor(cvec, mel.shape[0]) # [T, D]
        whisper = interpolate_tensor(whisper, mel.shape[0]) # [T, D]
        frame_len = mel.shape[0]

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
            mel = mel[start:end]
            rms = rms[start:end]
            f0 = f0[start:end]
            cvec = cvec[start:end]
            whisper = whisper[start:end]
            frame_len = self.max_frame_len

        result = dict(
            spk_id = spk_id,
            mel = mel,
            rms = rms,
            f0 = f0,
            cvec = cvec,
            whisper = whisper,
            frame_len = frame_len
        )

        return result


def load_svc_dataset(data_dir: str, meta_info_path: str, split = "train", max_frame_len = 256) -> SVCDataset:
    return SVCDataset(data_dir, meta_info_path, split, max_frame_len)


def collate_fn(batch):
    spk_ids = [item['spk_id'] for item in batch]
    mels = [item['mel'] for item in batch]
    rmss = [item['rms'] for item in batch]
    f0s = [item['f0'] for item in batch]
    cvecs = [item['cvec'] for item in batch]
    whispers = [item['whisper'] for item in batch]

    frame_lens = [item['frame_len'] for item in batch]

    # Pad sequences to max length
    mels_padded = pad_sequence(mels, batch_first=True)
    rmss_padded = pad_sequence(rmss, batch_first=True)
    f0s_padded = pad_sequence(f0s, batch_first=True)
    cvecs_padded = pad_sequence(cvecs, batch_first=True)
    whispers_padded = pad_sequence(whispers, batch_first=True)

    spk_ids = torch.cat(spk_ids)
    frame_len = torch.tensor(frame_lens)

    return {
        'spk_id': spk_ids,
        'mel': mels_padded,
        'rms': rmss_padded,
        'f0': f0s_padded,
        'cvec': cvecs_padded,
        'whisper': whispers_padded,
        'frame_len': frame_len
    }
