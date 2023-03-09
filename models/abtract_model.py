"""
@author: edenmyn
@email: edenmyn
@time: 2022/8/23 8:56
@DESC: 

"""
import torch
from pathlib import Path
from typing import Union


class AbstractModel(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
    
    def load(self, path: Union[str, Path]):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device), strict=False)

    def save(self, path: Union[str, Path]):
        torch.save(self.state_dict(), path, _use_new_zipfile_serialization=False)

    def get_step(self):
        return self.step.data.item()

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)
