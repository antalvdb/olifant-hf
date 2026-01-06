# Installation Instructions for Linux

This guide covers installing and testing the Olifant HuggingFace integration on Linux.

## Prerequisites

### System Requirements
- Linux (Ubuntu/Debian, Fedora/RHEL, or similar)
- Python 3.9 or higher (3.10+ recommended for python3-timbl)
- pip and virtualenv

### Install System Dependencies

#### Ubuntu/Debian
```bash
# Update package list
sudo apt-get update

# Install TiMBL command-line tool
sudo apt-get install -y timbl

# Install development tools (needed for python3-timbl compilation)
sudo apt-get install -y build-essential python3-dev libboost-python-dev

# Verify installation
timbl --version
```

#### Fedora/RHEL/CentOS
```bash
# Install TiMBL
sudo dnf install -y timbl

# Install development tools
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y python3-devel boost-python3-devel

# Verify installation
timbl --version
```

#### Arch Linux
```bash
# TiMBL may need to be compiled from source
# Install dependencies first
sudo pacman -S base-devel boost python

# Then compile TiMBL from: https://github.com/LanguageMachines/timbl
```

## Installation Steps

### 1. Extract the Package
```bash
cd ~
tar -xzf olifant-hf-package.tar.gz
cd olifant-hf-package
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
# Install base dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Try to install TiMBL Python bindings
# Note: This may fail if dependencies aren't perfect
pip install python3-timbl || echo "python3-timbl installation failed, will use CLI or mock backend"
```

### 4. Install the Package
```bash
# Install in development mode
pip install -e .
```

## Testing the Installation

### 1. Quick Test with Mock Backend
```bash
# This should always work
python demo_olifant_hf.py
```

Expected output:
- ✅ All 5 demos complete successfully
- Uses mock TiMBL backend (random predictions)
- Proves architecture is working

### 2. Test with Real TiMBL Models

If you have `.ibase` files:

```bash
# Copy your .ibase files to the package directory
cp /path/to/your/*.ibase .

# Modify test_olifant_hf.py to point to your model
# Then run:
python test_olifant_hf.py
```

### 3. Check Backend Status

Run this to see which TiMBL backend is active:

```python
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

from olifant_hf import OlifantForCausalLM, OlifantConfig

# This will print which backend is being used
config = OlifantConfig()
model = OlifantForCausalLM(config)
EOF
```

Possible outputs:
- `Using TiMBL Python bindings` ← Best option (if python3-timbl installed)
- `Using TiMBL CLI wrapper` ← Good option (if timbl command works)
- `Using MOCK TiMBL backend` ← Demo only

## Creating Fresh .ibase Files

To create compatible `.ibase` files on Linux:

### Option 1: Using Olifant Package (Recommended)

```bash
# Install olifant (requires Python 3.10+)
pip install olifant

# Tokenize your text
olifant-tok your_text_file.txt

# Create windowed training data
olifant-continuous-windowing your_text_file_tok 4

# Train with TiMBL
timbl -f your_text_file_tok.l4r0 -a4 +D -I your_model.l4r0.ibase

# Now you can use it!
```

### Option 2: Manual TiMBL Training

```bash
# Create training file (Columns format)
# Each line: feature1 feature2 feature3 feature4 label
cat > tiny_train.txt << 'EOF'
_ _ _ hello world
_ _ hello world !
_ hello world ! This
hello world ! This is
world ! This is great
EOF

# Train TiMBL model
timbl -f tiny_train.txt -F Columns -a4 +D -I tiny_model.ibase

# Test it
echo "_ _ _ hello ?" | timbl -i tiny_model.ibase -t - +v+db +v+di
```

## Troubleshooting

### Issue: `timbl: command not found`

**Solution**: Install TiMBL system package
```bash
# Ubuntu/Debian
sudo apt-get install timbl

# Or compile from source:
git clone https://github.com/LanguageMachines/timbl
cd timbl
bash bootstrap
./configure
make
sudo make install
```

### Issue: `python3-timbl` won't install

**Solution**: Use CLI wrapper or mock backend instead
- The CLI wrapper should work if `timbl` command is available
- Mock backend works for architecture demonstration

**Debug steps**:
```bash
# Check if timbl is available
which timbl
timbl --version

# Check Python version
python3 --version

# Check boost-python
ldconfig -p | grep boost_python
```

### Issue: Segmentation fault when loading .ibase files

**Cause**: Version mismatch between TiMBL version that created .ibase and current version

**Solution**: Recreate .ibase files with current TiMBL version
```bash
# Check TiMBL version
timbl --version

# Recreate .ibase from training data
timbl -f training_file.l4r0 -a4 +D -I new_model.ibase
```

### Issue: "Library not loaded" or "cannot open shared object"

**Solution**: Install missing dependencies
```bash
# Ubuntu/Debian
sudo apt-get install libxml2 libboost-all-dev

# Fedora/RHEL
sudo dnf install libxml2 boost-devel
```

## Performance Tips

### For Fast Testing
- Use window_size=4 (smaller context, faster)
- Use mock backend for architecture testing
- Use small .ibase files first

### For Production
- Use window_size=16 (better predictions)
- Install python3-timbl (best performance)
- Consider using timblserver for concurrent requests

## Next Steps

1. ✅ Run `python demo_olifant_hf.py` to verify installation
2. Create or obtain `.ibase` files compatible with your TiMBL version
3. Test with real models using `test_olifant_hf.py`
4. Integrate into your HuggingFace workflows

## Support

- Check `QUICKSTART_OLIFANT_HF.md` for usage examples
- Check `OLIFANT_HF_INTEGRATION_SUMMARY.md` for technical details
- Check `olifant_hf/README.md` for API documentation

## Expected Behavior on Linux

Linux should have BETTER compatibility than macOS because:
- ✅ Standard package managers (apt, dnf, etc.)
- ✅ Better library dependency management
- ✅ No Homebrew quirks
- ✅ TiMBL is primarily developed on Linux
- ✅ python3-timbl has better support

You should be able to get the full Python bindings working, which will give you the best performance!

Good luck! 🚀
