"""
Olifant (TiMBL-based) model configuration
"""

from transformers import PretrainedConfig
from typing import Optional


class OlifantConfig(PretrainedConfig):
    """
    Configuration class for Olifant models.

    Olifant uses memory-based learning (TiMBL) instead of neural networks,
    storing instance bases (.ibase files) rather than weight matrices.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the tokenizer. Used for converting distributions to logits.
        window_size (`int`, *optional*, defaults to 4):
            Context window size (number of previous tokens used for prediction).
            Common values: 4 (l4r0), 16 (l16r0).
        timbl_options (`str`, *optional*, defaults to "-a4 +D"):
            TiMBL algorithm options. Common options:
            - `-a0`: IB1 (exact matching)
            - `-a1`: IGTREE (decision tree)
            - `-a4`: IB1 with modified value difference metric
            - `+D`: Enable class distribution in output
        pad_token (`str`, *optional*, defaults to "_"):
            Token used for padding shorter sequences to window_size.
        model_prefix (`str`, *optional*):
            Base filename for .ibase and .wgt files (without extension).
        ibase_path (`str`, *optional*):
            Full path to the .ibase file. Takes precedence over model_prefix if set.
        normalize_distributions (`bool`, *optional*, defaults to True):
            Whether to normalize class distributions when converting to logits.
    """

    model_type = "olifant"

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 default
        window_size: int = 4,
        timbl_options: str = "-a4 +D",
        pad_token: str = "_",
        model_prefix: Optional[str] = None,
        ibase_path: Optional[str] = None,
        normalize_distributions: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.timbl_options = timbl_options
        self.pad_token = pad_token
        self.model_prefix = model_prefix
        self.ibase_path = ibase_path
        self.normalize_distributions = normalize_distributions
