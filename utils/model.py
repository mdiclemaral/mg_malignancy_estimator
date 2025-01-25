import torch
import numpy as np

from utils.mass import MassSegmentation
from utils.roi import ROIextraction

def load_model(checkpoint: bytes, device: torch.device) -> MassSegmentation:
    model = MassSegmentation(device)
    model.load(checkpoint)
    return model

def load_roi(checkpoint: bytes, device: torch.device) -> ROIextraction:
    model = ROIextraction(device)
    model.load(checkpoint)
    return model

def load_checkpoint_bytes(path: str) -> bytes:
    with open(path, "rb") as file:
        return file.read()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/mass.pt"
    
    checkpoint_bytes = load_checkpoint_bytes(model_path)
    model = load_model(checkpoint_bytes, device)
    