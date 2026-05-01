import torch
from models import vit_base, vit_large
import re


def load_simdino_state_dict(weights_path):
    state_dict = torch.load(weights_path)
    state_dict = state_dict["teacher"]
    state_dict = {k.removeprefix("backbone."): v for k, v in state_dict.items() if k.startswith("backbone.")}
    state_dict = {remap_key(k): v for k, v in state_dict.items()}
    return state_dict


def remap_key(k):
    """Remap keys in the state dictionary."""
    return re.sub(r"blocks\.\d+\.(?=\d+)", "blocks.", k)


def initialize_encoder(arch='base'):
    
    img_size = 224
    patch_size = 16

    if arch == 'vitb':
        encoder = vit_base(patch_size=patch_size, img_size=img_size, init_values=0.1, block_chunks=0, num_register_tokens=4)
    elif arch == 'vitl':
        encoder = vit_large(patch_size=patch_size, img_size=img_size, init_values=0.1, block_chunks=0, num_register_tokens=4)

    return encoder


def get_encoder(args):

    arch = 'vitb' if 'vitb' in args.ckpt else 'vitl'
    encoder = initialize_encoder(arch=arch)
    state_dict = load_simdino_state_dict(args.ckpt)
    encoder.load_state_dict(state_dict, strict=True)

    encoder = encoder.to(args.device)

    return encoder

