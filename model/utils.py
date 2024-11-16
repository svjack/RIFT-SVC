import os
import random
from typing import Any
from jaxtyping import Int, Bool

import torch


def seed_everything(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# helpers

def exists(v: Any) -> bool:
    return v is not None

def default(v: Any, d: Any) -> Any:
    return v if exists(v) else d

# tensor helpers

def lens_to_mask(
    t: Int[torch.Tensor, "b"],
    length: int | None = None
) -> Bool[torch.Tensor, "b n"]: 

    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device = t.device)
    return seq < t[..., None]