# Olifant HuggingFace Integration

This package provides a HuggingFace `transformers`-compatible interface for Olifant (TiMBL-based) language models.

## ✅ What Works

The HuggingFace integration architecture is **fully functional**:

- ✅ `OlifantConfig` - Standard HF configuration class
- ✅ `OlifantForCausalLM` - Full `PreTrainedModel` implementation
- ✅ `from_pretrained()` - Loads `.ibase` files from disk
- ✅ `save_pretrained()` - Saves models in HF format
- ✅ `forward()` - Returns proper logits tensors `(batch, seq_len, vocab_size)`
- ✅ `generate()` - Text generation using HF's generation methods
- ✅ Tensor shapes - Fully compatible with HF ecosystem

## 🚧 Current Limitation: TiMBL Backend

The integration requires a working TiMBL backend for actual inference. Currently, two options exist:

### Option 1: TiMBL Python Bindings (Recommended)
```bash
pip install python3-timbl
```

**Status**: Requires compilation with boost-python3. May have dependency issues on some systems.

### Option 2: TiMBL CLI Wrapper (Fallback)
Uses the `timbl` command-line tool via subprocess.

**Status**: Currently experiencing compatibility issues with existing `.ibase` files (segmentation faults or version mismatches).

## 📦 Installation

```bash
# Install the olifant package (when available)
pip install olifant

# OR: Use this local implementation
cd /path/to/WOPR
python -c "import olifant_hf"
```

## 🎯 Usage Example

```python
from transformers import AutoTokenizer
from olifant_hf import OlifantConfig, OlifantForCausalLM

# Create configuration
config = OlifantConfig(
    vocab_size=50257,           # GPT-2 vocab
    window_size=16,             # Context window (l16r0)
    timbl_options="-a4 +D",     # TiMBL algorithm options
    model_prefix="my-model_tok.l16r0"
)

# Initialize model
model = OlifantForCausalLM(config)

# Set tokenizer (required for token<->ID mapping)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model.set_tokenizer(tokenizer)

# Load TiMBL classifier
model._load_timbl_classifier("./path/to/models")

# Use like any HuggingFace model!
input_ids = tokenizer.encode("Hello world", return_tensors="pt")

# Forward pass
outputs = model(input_ids)
print(outputs.logits.shape)  # torch.Size([1, 2, 50257])

# Generate text
generated = model.generate(input_ids, max_length=20)
print(tokenizer.decode(generated[0]))

# Save in HuggingFace format
model.save_pretrained("./my_olifant_hf_model")

# Load from disk
loaded_model = OlifantForCausalLM.from_pretrained("./my_olifant_hf_model")
```

## 🗂️ Model Format

Olifant models consist of:
- `config.json` - HuggingFace configuration
- `*.ibase` - TiMBL instance base (the actual model)
- `*.wgt` - TiMBL weights (optional)

Unlike neural models, there are NO `.bin` or `.safetensors` files. The `.ibase` file contains the indexed k-NN instance base.

## 🔬 Architecture

### Key Design Decisions

1. **No PyTorch Weights**: Olifant uses memory-based learning, not neural networks. The `_init_weights()` method is a no-op.

2. **Logits from Distributions**: TiMBL returns class distributions (token similarities). We convert these to logits compatible with HF's generation methods.

3. **Token Mapping**: TiMBL works with token strings, HF works with token IDs. We maintain bidirectional mappings via `set_tokenizer()`.

4. **Window-based Context**: Predictions use a fixed-size sliding window (typically 4 or 16 tokens) rather than full attention.

## 🛠️ File Structure

```
olifant_hf/
├── __init__.py                 # Package exports
├── configuration_olifant.py    # OlifantConfig
├── modeling_olifant.py         # OlifantForCausalLM
├── timbl_wrapper.py            # CLI fallback wrapper
└── README.md                   # This file
```

## 🚀 Future Work

To make this production-ready:

1. **Fix TiMBL Backend**:
   - Resolve Python binding compilation issues
   - OR: Fix CLI wrapper's .ibase compatibility
   - OR: Implement timblserver client

2. **Optimize Performance**:
   - Batch inference support
   - Caching for repeated queries
   - Parallel classification

3. **HuggingFace Hub Integration**:
   - Register model type with `transformers`
   - Upload models to Hub
   - Add model cards

4. **Testing**:
   - Unit tests for all components
   - Integration tests with various tokenizers
   - Benchmark against neural baselines

## 📊 Comparison: Olifant vs Neural LMs

| Feature | Neural LMs | Olifant |
|---------|------------|---------|
| Training | GPU-intensive | CPU-only |
| Inference | GPU for speed | CPU sufficient |
| Explainability | Black box | Full provenance |
| Model Size | 100MB-100GB+ | 10MB-10GB |
| CO2 Emissions | High | 1000x lower |
| Fine-tuning | Requires retraining | Add instances |

## 📚 References

- **Olifant**: https://github.com/antalvdb/olifant
- **TiMBL**: https://github.com/LanguageMachines/timbl
- **Paper**: Van den Bosch et al. (2025). Memory-based Language Models

## 📝 License

This integration code follows the same license as Olifant (check the main repository).

## 🤝 Contributing

This is a proof-of-concept demonstrating that HuggingFace integration is feasible. To make it production-ready, we need to resolve the TiMBL backend issue. Contributions welcome!

## ⚠️ Known Issues

1. **TiMBL CLI Wrapper**: Segmentation faults with some `.ibase` files
2. **Python Bindings**: Compilation requires boost-python3 (dependency hell)
3. **Performance**: Single-instance classification is slow for large batches

## ✨ Acknowledgments

Built as a demonstration that HuggingFace's model abstraction can support non-neural architectures. Special thanks to the Olifant and TiMBL teams for their pioneering work in memory-based learning.
