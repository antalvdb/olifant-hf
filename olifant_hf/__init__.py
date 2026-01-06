"""
Olifant: TiMBL-based language models for HuggingFace transformers
"""

from .configuration_olifant import OlifantConfig
from .modeling_olifant import OlifantForCausalLM

__all__ = ["OlifantConfig", "OlifantForCausalLM"]
