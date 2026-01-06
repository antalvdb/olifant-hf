"""
Test script for Olifant HuggingFace integration
"""

import pytest
import torch
from transformers import AutoTokenizer
from olifant_hf import OlifantConfig, OlifantForCausalLM


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Fixture to create and load model and tokenizer once per module."""
    # Create config
    config = OlifantConfig(
        vocab_size=50257,
        window_size=4,
        timbl_options="-a4 +D",
        model_prefix="edufineweb_train_000001-100k.tok.l4r0"
    )

    # Initialize model
    model = OlifantForCausalLM(config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model.set_tokenizer(tokenizer)

    # Load TiMBL classifier
    model._load_timbl_classifier(".")

    return model, tokenizer


@pytest.fixture
def model(model_and_tokenizer):
    """Fixture to get model."""
    return model_and_tokenizer[0]


@pytest.fixture
def tokenizer(model_and_tokenizer):
    """Fixture to get tokenizer."""
    return model_and_tokenizer[1]


def test_basic_loading(model, tokenizer):
    """Test basic model loading"""
    assert model is not None
    assert tokenizer is not None
    assert model.config.vocab_size == 50257
    assert model.config.window_size == 4
    assert model.timbl_classifier is not None


def test_forward_pass(model, tokenizer):
    """Test forward pass"""
    # Prepare input
    text = "Hello, how are"
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)

    # Check output shape
    batch_size, seq_len = input_ids.shape
    assert outputs.logits.shape == (batch_size, seq_len, model.config.vocab_size)

    # Check we can get a prediction
    next_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    assert 0 <= next_token_id < model.config.vocab_size


def test_generation(model, tokenizer):
    """Test text generation"""
    prompt = "Een aantal kleuren:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = input_ids.shape[1]

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=prompt_len + 10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Check output
    assert output_ids.shape[0] == 1  # batch size
    assert output_ids.shape[1] > prompt_len  # generated something

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assert len(generated_text) > len(prompt)


def test_save_and_load(model, tokenizer, tmp_path):
    """Test save_pretrained and from_pretrained"""
    save_dir = tmp_path / "test_model"

    # Save
    model.save_pretrained(str(save_dir))
    assert (save_dir / "config.json").exists()

    # Load
    loaded_model = OlifantForCausalLM.from_pretrained(str(save_dir))
    loaded_model.set_tokenizer(tokenizer)

    # Test that loaded model works
    text = "Test input"
    input_ids = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        outputs = loaded_model(input_ids)

    batch_size, seq_len = input_ids.shape
    assert outputs.logits.shape == (batch_size, seq_len, loaded_model.config.vocab_size)


def main():
    """Run all tests directly (without pytest)"""
    import tempfile
    from pathlib import Path

    print("\n" + "=" * 80)
    print("OLIFANT HUGGINGFACE INTEGRATION TEST SUITE")
    print("=" * 80)

    try:
        # Setup: Load model and tokenizer
        print("\nSetting up model and tokenizer...")
        model, tokenizer = model_and_tokenizer()

        # Test 1: Basic loading
        print("\nTest 1: Basic loading...")
        test_basic_loading(model, tokenizer)
        print("✓ Passed")

        # Test 2: Forward pass
        print("\nTest 2: Forward pass...")
        test_forward_pass(model, tokenizer)
        print("✓ Passed")

        # Test 3: Generation
        print("\nTest 3: Generation...")
        test_generation(model, tokenizer)
        print("✓ Passed")

        # Test 4: Save and load
        print("\nTest 4: Save and load...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_save_and_load(model, tokenizer, Path(tmp_dir))
        print("✓ Passed")

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
