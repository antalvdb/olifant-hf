"""
Olifant model implementation for HuggingFace transformers
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import json
import shutil

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from huggingface_hub import hf_hub_download, snapshot_download

# Try to import TiMBL backends in order of preference
TIMBL_MODE = None
timbl = None

# Try 1: Python bindings (best performance)
try:
    import timbl
    TIMBL_MODE = "python_bindings"
    print("Using TiMBL Python bindings")
except ImportError:
    pass

# Try 2: Mock backend (for demonstration only!)
if TIMBL_MODE is None:
    try:
        from mock_timbl import MockTimblClassifier as timbl_MockClassifier
    except ImportError:
        from .mock_timbl import MockTimblClassifier as timbl_MockClassifier
    TIMBL_MODE = "mock"
    print("⚠️  Using MOCK TiMBL backend - predictions are random!")
    print("   For real inference, install python3-timbl")

    class TimblModule:
        TimblClassifier = timbl_MockClassifier

    timbl = TimblModule()

try:
    from configuration_olifant import OlifantConfig
except ImportError:
    from .configuration_olifant import OlifantConfig


class OlifantForCausalLM(PreTrainedModel):
    """
    Olifant model for causal language modeling using TiMBL (memory-based learning).

    Unlike traditional neural LMs, Olifant stores training instances in an indexed
    instance base (.ibase file) and uses k-nearest neighbors for prediction.
    """

    config_class = OlifantConfig
    base_model_prefix = "olifant"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(self, config: OlifantConfig):
        super().__init__(config)
        self.config = config

        # TiMBL classifier (will be loaded later)
        self.timbl_classifier: Optional[timbl.TimblClassifier] = None

        # Cache for token-to-id mapping (populated from tokenizer)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Dummy parameter for HuggingFace device detection
        # (memory-based models have no real parameters, but HF needs this for generate())
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Post-init
        self.post_init()

    def _init_weights(self, module):
        """No PyTorch weights to initialize for memory-based model."""
        pass

    def can_generate(self) -> bool:
        """This model supports generation."""
        return True

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for generation."""
        return {"input_ids": input_ids}

    def _load_timbl_classifier(self, model_path: str):
        """
        Load the TiMBL classifier from .ibase file.

        Args:
            model_path: Path to the directory containing .ibase file, or direct path to .ibase
        """
        ibase_path = None
        model_prefix = None
        
        # Check if this is a HuggingFace Hub model ID (contains / but is not a local path)
        is_hf_hub = "/" in model_path and not os.path.exists(model_path)
        
        if is_hf_hub:
            # Download from HuggingFace Hub
            print(f"Downloading model files from HuggingFace Hub: {model_path}")
            
            # Download the entire repo to get all files
            local_dir = snapshot_download(
                repo_id=model_path,
                allow_patterns=["*.ibase", "*.wgt", "config.json"]
            )
            model_path = local_dir
        
        # Determine the .ibase file path
        if model_path.endswith('.ibase'):
            ibase_path = model_path
            model_prefix = ibase_path[:-6]  # Remove .ibase extension
        elif self.config.ibase_path:
            ibase_path = self.config.ibase_path
            model_prefix = ibase_path[:-6] if ibase_path.endswith('.ibase') else ibase_path
        elif self.config.model_prefix:
            model_prefix = os.path.join(model_path, self.config.model_prefix)
            ibase_path = model_prefix + '.ibase'
        else:
            # Look for any .ibase file in the directory
            ibase_files = [f for f in os.listdir(model_path) if f.endswith('.ibase')]
            if not ibase_files:
                raise FileNotFoundError(f"No .ibase file found in {model_path}")
            ibase_path = os.path.join(model_path, ibase_files[0])
            model_prefix = ibase_path[:-6]

        # Check if file exists (skip for mock backend)
        if TIMBL_MODE != "mock" and not os.path.exists(ibase_path):
            raise FileNotFoundError(f"Instance base not found: {ibase_path}")

        print(f"Loading TiMBL classifier from: {ibase_path}")

        # Initialize and load TiMBL classifier
        self.timbl_classifier = timbl.TimblClassifier(
            model_prefix,
            self.config.timbl_options,
            format="Tabbed"
        )
        self.timbl_classifier.load()

        print(f"TiMBL classifier loaded successfully with options: {self.config.timbl_options}")

    def set_tokenizer(self, tokenizer):
        """
        Set the tokenizer and build token<->id mappings.

        This is necessary because TiMBL works with token strings,
        but HuggingFace models work with token IDs.
        """
        self.tokenizer = tokenizer

        # Build mapping from tokens to IDs
        if hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            self.token_to_id = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        else:
            # Fallback: build mapping on-the-fly
            self.token_to_id = {}
            self.id_to_token = {}

        # Update vocab size if needed
        if len(self.token_to_id) > 0:
            self.config.vocab_size = max(self.config.vocab_size, len(self.token_to_id))

    def _ids_to_tokens(self, input_ids: torch.Tensor) -> list:
        """Convert token IDs to token strings."""
        if hasattr(self, 'tokenizer'):
            return self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        else:
            # Use cached mapping
            return [self.id_to_token.get(id.item(), str(id.item())) for id in input_ids]

    def _token_to_id(self, token: str) -> int:
        """Convert token string to ID."""
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            return self.tokenizer.convert_tokens_to_ids(token)
        return self.token_to_id.get(token, 0)

    def _pad_prompt(self, tokens: list, max_len: Optional[int] = None) -> list:
        """Pad or trim token list to window size."""
        if max_len is None:
            max_len = self.config.window_size

        if len(tokens) < max_len:
            # Pad left with pad_token
            tokens = [self.config.pad_token] * (max_len - len(tokens)) + tokens
        else:
            # Take last max_len tokens
            tokens = tokens[-max_len:]

        return tokens

    def _distribution_to_logits(self, distribution, distance: float) -> torch.Tensor:
        """
        Convert TiMBL's class distribution to HuggingFace-style logits.

        Args:
            distribution: TiMBL distribution - either a dict (Python bindings) or
                         string (CLI wrapper) like "token1 0.7, token2 0.3"
            distance: Distance to nearest neighbors

        Returns:
            Logits tensor of shape (vocab_size,)
        """
        logits = torch.full((self.config.vocab_size,), -100.0)  # Very low logit for unseen tokens

        if not distribution or distribution == "?":
            return logits

        try:
            # Handle dict format from TiMBL Python bindings
            if isinstance(distribution, dict):
                for token, score in distribution.items():
                    token_id = self._token_to_id(token)
                    if token_id < self.config.vocab_size:
                        logits[token_id] = float(score) if score > 0 else -100.0
            else:
                # Parse distribution string from CLI wrapper
                # Format: "token1 score1, token2 score2, ..."
                pairs = distribution.split(',')
                for pair in pairs:
                    parts = pair.strip().split()
                    if len(parts) >= 2:
                        token = parts[0]
                        score = float(parts[1])

                        # Convert token to ID
                        token_id = self._token_to_id(token)

                        if token_id < self.config.vocab_size:
                            # Convert score to logit (log probability)
                            # TiMBL scores are similarities/frequencies, not probabilities
                            logits[token_id] = score if score > 0 else -100.0

        except Exception as e:
            print(f"Warning: Failed to parse distribution '{distribution}': {e}")

        return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Mask of shape (batch_size, sequence_length)
            labels: Labels for language modeling loss
            return_dict: Whether to return ModelOutput object

        Returns:
            CausalLMOutputWithPast with logits of shape (batch_size, sequence_length, vocab_size)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.timbl_classifier is None:
            raise RuntimeError("TiMBL classifier not loaded. Call from_pretrained() first.")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize logits tensor
        logits = torch.zeros(batch_size, seq_len, self.config.vocab_size, device=device)

        # Process each sequence in the batch
        for batch_idx in range(batch_size):
            sequence_ids = input_ids[batch_idx]

            # Convert IDs to tokens
            tokens = self._ids_to_tokens(sequence_ids)

            # For each position, predict next token using sliding window
            for pos in range(seq_len):
                # Get context window ending at current position
                context_tokens = tokens[max(0, pos - self.config.window_size):pos]

                # Pad context to window size
                padded_context = self._pad_prompt(context_tokens, self.config.window_size)

                try:
                    # Query TiMBL classifier
                    classlabel, distribution, distance = self.timbl_classifier.classify(padded_context)

                    # Convert distribution to logits
                    position_logits = self._distribution_to_logits(distribution, distance)
                    logits[batch_idx, pos] = position_logits

                except Exception as e:
                    print(f"Warning: Classification failed at position {pos}: {e}")
                    # Leave logits as zeros (uniform distribution)
                    pass

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load model from directory containing config.json and .ibase files.

        Args:
            pretrained_model_name_or_path: Path to model directory or HuggingFace model ID
        """
        # Load config
        config = kwargs.pop("config", None)
        if config is None:
            config = OlifantConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Initialize model
        model = cls(config)

        # Load TiMBL classifier
        model._load_timbl_classifier(pretrained_model_name_or_path)

        return model

    def save_pretrained(
        self,
        save_directory: str,
        **kwargs
    ):
        """
        Save model to directory.

        Saves:
        - config.json (HuggingFace config)
        - *.ibase (TiMBL instance base)
        - *.wgt (TiMBL weights, if present)
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_directory)

        # Copy .ibase and .wgt files
        if self.timbl_classifier:
            source_prefix = self.timbl_classifier.fileprefix
            ibase_src = source_prefix + '.ibase'
            wgt_src = source_prefix + '.wgt'

            if os.path.exists(ibase_src):
                # Determine destination filename
                dest_basename = os.path.basename(source_prefix)
                ibase_dest = os.path.join(save_directory, dest_basename + '.ibase')
                shutil.copy2(ibase_src, ibase_dest)
                print(f"Saved instance base to: {ibase_dest}")

                # Update config with new model_prefix
                self.config.model_prefix = dest_basename
                self.config.save_pretrained(save_directory)

            if os.path.exists(wgt_src):
                wgt_dest = os.path.join(save_directory, dest_basename + '.wgt')
                shutil.copy2(wgt_src, wgt_dest)
                print(f"Saved weights to: {wgt_dest}")

        print(f"Model saved to: {save_directory}")
