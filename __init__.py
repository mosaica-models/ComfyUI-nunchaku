from loguru import logger
from huggingface_hub import hf_hub_download as orig_hf_hub_download
def patched_hf_hub_download(model_id, filename, revision=None, cache_dir=None, local_dir=None, *args, **kwargs):
    logger.info(f"hf download {model_id} {filename}")
    if "EVA-CLIP" in model_id:
        local_dir = "/root/.cache/eva-clip"
    return orig_hf_hub_download(
        model_id, filename, revision=revision, cache_dir=cache_dir, local_dir=local_dir, *args, **kwargs
    )
import huggingface_hub
huggingface_hub.hf_hub_download = patched_hf_hub_download

# only import if running as a custom node
from .nodes.lora import NunchakuFluxLoraLoader
from .nodes.models import (
    NunchakuFluxDiTLoader,
    NunchakuPulidApply,
    NunchakuPulidLoader,
    NunchakuTextEncoderLoader,
    NunchakuTextEncoderLoaderV2,
)
from .nodes.preprocessors import FluxDepthPreprocessor
from .nodes.tools import NunchakuModelMerger

NODE_CLASS_MAPPINGS = {
    "NunchakuFluxDiTLoader": NunchakuFluxDiTLoader,
    "NunchakuTextEncoderLoader": NunchakuTextEncoderLoader,
    "NunchakuTextEncoderLoaderV2": NunchakuTextEncoderLoaderV2,
    "NunchakuFluxLoraLoader": NunchakuFluxLoraLoader,
    "NunchakuDepthPreprocessor": FluxDepthPreprocessor,
    "NunchakuPulidApply": NunchakuPulidApply,
    "NunchakuPulidLoader": NunchakuPulidLoader,
    "NunchakuModelMerger": NunchakuModelMerger,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
