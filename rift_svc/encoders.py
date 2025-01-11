import torch
from torch import nn
from typing import Optional

from transformers import HubertModel
import transformers.models.whisper.modeling_whisper as whisper


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class WhisperEncoder(whisper.WhisperEncoder):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config = whisper.WhisperConfig.from_pretrained(*args, **kwargs)
        encoder = cls(config)
        encoder_state_dict = whisper.WhisperModel.from_pretrained(*args, **kwargs).encoder.state_dict()
        encoder.load_state_dict(encoder_state_dict)
        return encoder