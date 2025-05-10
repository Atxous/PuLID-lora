import os
from dataclasses import dataclass
import re
import time
import requests
from pathlib import Path
from urllib.parse import urlparse

import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
#from imwatermark import WatermarkEncoder
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder

MODEL_CACHE = "models"


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int

@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str
    ae_path: str
    repo_id: str
    repo_flow: str
    repo_ae: str


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path='models/flux1-dev.safetensors',
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path='models/ae.safetensors',
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(name: str, device: str = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if ckpt_path is None or not os.path.exists(ckpt_path):
        if configs[name].repo_id is not None and configs[name].repo_flow is not None and hf_download:
            print(f"Downloading {configs[name].repo_flow} from {configs[name].repo_id}")
            ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    with torch.device(device):
        model = Flux(configs[name].params).to(torch.bfloat16)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(ckpt_path, device=str(device))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print_load_warning(missing, unexpected)
    return model


def load_t5(device: str = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("XLabs-AI/xflux_text_encoders", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str = "cuda", hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if ckpt_path is None or not os.path.exists(ckpt_path):
        if configs[name].repo_id is not None and configs[name].repo_ae is not None and hf_download:
            print(f"Downloading {configs[name].repo_ae} from {configs[name].repo_id}")
            ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)
        else:
            raise FileNotFoundError(f"Autoencoder checkpoint file not found: {ckpt_path}")

    # Loading the autoencoder
    print("Init AE")
    with torch.device(device):
        ae = AutoEncoder(configs[name].ae_params)

    print("Loading AE checkpoint")
    sd = load_sft(ckpt_path, device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False)
    print_load_warning(missing, unexpected)
    return ae


# class WatermarkEmbedder:
#     def __init__(self, watermark):
#         self.watermark = watermark
#         self.num_bits = len(WATERMARK_BITS)
#         self.encoder = WatermarkEncoder()
#         self.encoder.set_watermark("bits", self.watermark)

#     def __call__(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Adds a predefined watermark to the input image

#         Args:
#             image: ([N,] B, RGB, H, W) in range [-1, 1]

#         Returns:
#             same as input but watermarked
#         """
#         image = 0.5 * image + 0.5
#         squeeze = len(image.shape) == 4
#         if squeeze:
#             image = image[None, ...]
#         n = image.shape[0]
#         image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
#         # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
#         # watermarking libary expects input as cv2 BGR format
#         for k in range(image_np.shape[0]):
#             image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
#         image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(
#             image.device
#         )
#         image = torch.clamp(image / 255, min=0.0, max=1.0)
#         if squeeze:
#             image = image[0]
#         image = 2 * image - 1
#         return image


# # A fixed 48-bit message that was choosen at random
# WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# # bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
# WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
# embed_watermark = WatermarkEmbedder(WATERMARK_BITS)


def download_from_civitai(url: str) -> str:
    """Download a LoRA from CivitAI."""
    try:
        # Extract model ID from URL
        model_id = re.search(r'civitai\.com/models/(\d+)', url).group(1)
        api_url = f"https://civitai.com/api/v1/models/{model_id}"
        
        # Get model info
        response = requests.get(api_url)
        response.raise_for_status()
        model_info = response.json()
        
        # Get download URL from the latest version
        download_url = model_info["modelVersions"][0]["files"][0]["downloadUrl"]
        
        # Download the file
        local_path = os.path.join(MODEL_CACHE, f"civitai_{model_id}.safetensors")
        response = requests.get(download_url)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            f.write(response.content)
            
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download from CivitAI: {str(e)}")

def download_from_replicate(model_path: str) -> str:
    """Download a LoRA from Replicate."""
    try:
        # Parse model path (owner/model or owner/model/version)
        parts = model_path.split('/')
        if len(parts) == 2:
            owner, model = parts
            version = "latest"  # You might want to implement a way to get latest version
        else:
            owner, model, version = parts
            
        # Construct download URL (you'll need to implement this based on Replicate's API)
        download_url = f"https://replicate.delivery/{owner}/{model}/{version}/lora.safetensors"
        
        # Download the file
        local_path = os.path.join(MODEL_CACHE, f"replicate_{owner}_{model}_{version}.safetensors")
        response = requests.get(download_url)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            f.write(response.content)
            
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download from Replicate: {str(e)}")

def download_from_url(url: str) -> str:
    """Download a LoRA from a direct URL."""
    try:
        local_path = os.path.join(MODEL_CACHE, os.path.basename(url))
        response = requests.get(url)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            f.write(response.content)
            
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download from URL: {str(e)}")

def resolve_lora_path(lora_path: str) -> str:
    """
    Resolve various LoRA path formats to a local file path.
    Supports:
    - Local files
    - Direct URLs (http(s)://...)
    - CivitAI URLs (civitai.com/...)
    - Replicate models (owner/model or owner/model/version)
    """
    if not lora_path:
        return None
        
    # If it's already a local file
    if os.path.exists(lora_path):
        return lora_path
        
    # Create cache directory if it doesn't exist
    os.makedirs(MODEL_CACHE, exist_ok=True)
    
    # Parse URL if it's a URL
    try:
        parsed = urlparse(lora_path)
        if parsed.scheme in ['http', 'https']:
            if 'civitai.com' in parsed.netloc:
                return download_from_civitai(lora_path)
            else:
                return download_from_url(lora_path)
    except Exception:
        pass
    
    # If it's in the format owner/model[/version], treat as Replicate model
    if re.match(r'^[\w-]+/[\w-]+(?:/[\w-]+)?$', lora_path):
        return download_from_replicate(lora_path)
        
    raise ValueError(f"Invalid LoRA path format: {lora_path}")

def load_lora(model: Flux, lora_path: str, alpha: float = 1.0, device: str = "cuda"):
    """
    Load and apply LoRA weights to the model.
    
    Args:
        model: The base model to apply LoRA weights to
        lora_path: Path to the LoRA. Can be:
            - Local file path
            - Direct URL to .safetensors file
        alpha: The weight to apply the LoRA (default: 1.0)
        device: Device to load the LoRA weights to (default: "cuda")
    
    Returns:
        The model with LoRA weights applied
    """
    try:
        # Resolve the LoRA path to a local file
        local_path = resolve_lora_path(lora_path)
        if not local_path:
            print(f"[LoRA] No valid local path found for {lora_path}")
            return model
            
        print(f"[LoRA] Loading weights from: {local_path}")
        
        # Load LoRA weights
        lora_state_dict = load_sft(local_path, device=str(device))
        print(f"[LoRA] Loaded {len(lora_state_dict)} keys from LoRA file")
        
        # Get original state dict
        orig_state_dict = model.state_dict()
        print(f"[LoRA] Base model has {len(orig_state_dict)} keys")
        
        # Find all LoRA keys
        lora_keys = [k for k in lora_state_dict.keys() if 'lora_' in k]
        print(f"[LoRA] Found {len(lora_keys)} LoRA keys")
        
        if not lora_keys:
            print("[LoRA] Warning: No LoRA keys found in the loaded weights")
            return model
            
        # Print some example LoRA keys for debugging
        print("[LoRA] Example LoRA keys:", lora_keys[:5])
            
        # Apply LoRA weights
        modified_keys = []
        
        # Group LoRA keys by their base name (without lora_A/B)
        lora_groups = {}
        for key in lora_keys:
            base_key = key.replace('.lora_A.weight', '').replace('.lora_B.weight', '')
            if base_key not in lora_groups:
                lora_groups[base_key] = {'A': None, 'B': None}
            if 'lora_A' in key:
                lora_groups[base_key]['A'] = key
            elif 'lora_B' in key:
                lora_groups[base_key]['B'] = key
        
        print(f"[LoRA] Found {len(lora_groups)} LoRA weight groups")
        
        # Define key mapping for transformer blocks
        key_mapping = {
            'transformer.': '',  # Remove transformer prefix
            'single_transformer_blocks': 'single_blocks',
            'double_transformer_blocks': 'double_blocks',
            'attn.to_q': 'img_attn.qkv',
            'attn.to_k': 'img_attn.qkv',
            'attn.to_v': 'img_attn.qkv',
            'attn.to_out.0': 'img_attn.proj',
            'mlp.fc1': 'img_mlp.0',
            'mlp.fc2': 'img_mlp.2',
            'norm1': 'img_norm1',
            'norm2': 'img_norm2'
        }
        
        # Try to map transformer keys to FLUX model keys
        for base_key, weights in lora_groups.items():
            if weights['A'] is None or weights['B'] is None:
                continue
                
            # Try to map the transformer key to a FLUX key
            flux_key = base_key
            for old, new in key_mapping.items():
                flux_key = flux_key.replace(old, new)
            
            # Remove any trailing dots
            flux_key = flux_key.rstrip('.')
            
            if flux_key in orig_state_dict:
                print(f"[LoRA] Applying LoRA to {flux_key}")
                # Compute the merged weights
                weight_A = lora_state_dict[weights['A']].float()
                weight_B = lora_state_dict[weights['B']].float()
                
                # Merge weights: original + (B × A) × alpha
                delta = (weight_B @ weight_A) * alpha
                orig_weight = orig_state_dict[flux_key].float()
                
                # Handle different shapes for QKV
                if 'qkv' in flux_key:
                    # FLUX combines Q, K, V into a single tensor
                    # Expand delta to match QKV shape if needed
                    if delta.shape[0] != orig_weight.shape[0]:
                        # Assuming the order is Q, K, V
                        delta = torch.cat([delta] * 3, dim=0)
                
                # Add the delta and convert back to original dtype
                orig_state_dict[flux_key] = (orig_weight + delta).to(orig_weight.dtype)
                modified_keys.append(flux_key)
            else:
                print(f"[LoRA] Could not find matching key for {base_key}")
                print(f"[LoRA] Attempted to map to: {flux_key}")
                # Print the actual keys in the model that are similar
                similar_keys = [k for k in orig_state_dict.keys() if any(part in k for part in flux_key.split('.'))]
                if similar_keys:
                    print(f"[LoRA] Similar keys in model: {similar_keys[:5]}")
        
        print(f"[LoRA] Modified {len(modified_keys)} keys in the model")
        if modified_keys:
            print("[LoRA] First few modified keys:", modified_keys[:5])
        
        # Load the merged weights back into the model
        missing, unexpected = model.load_state_dict(orig_state_dict, strict=False)
        if missing:
            print(f"[LoRA] Missing keys after applying LoRA: {missing}")
        if unexpected:
            print(f"[LoRA] Unexpected keys after applying LoRA: {unexpected}")
        
        return model
        
    except Exception as e:
        print(f"[LoRA] Error loading LoRA from {lora_path}: {str(e)}")
        import traceback
        print("[LoRA] Full traceback:", traceback.format_exc())
        return model
