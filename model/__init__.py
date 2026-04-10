from .unet import UNet2D
from .crn import CRN
from .dccrn import DCCRN

MODEL_REGISTRY = {
    'unet': UNet2D,
    'crn': CRN,
    'dccrn': DCCRN,
}

def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)