"""
Test script for Olifant HuggingFace integration
"""

import torch
from transformers import AutoTokenizer
from olifant_hf import OlifantConfig, OlifantForCausalLM


def test_basic_loading():
    """Test basic model loading"""
    print("=" * 80)
    print("TEST 1: Basic Model Loading")
    print("=" * 80)

    # Create config
    config = OlifantConfig(
        vocab_size=50257,
        window_size=4,
        timbl_options="-a4 +D",
        model_prefix="edufineweb_train_000001-100k.tok.l4r0"
    )

    print(f"Config created: {config}")
    print(f"  - vocab_size: {config.vocab_size}")
    print(f"  - window_size: {config.window_size}")
    print(f"  - timbl_options: {config.timbl_options}")

    # Initialize model
    print("\nInitializing model...")
    model = OlifantForCausalLM(config)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model.set_tokenizer(tokenizer)

    # Load TiMBL classifier
    print("\nLoading TiMBL classifier...")
    model._load_timbl_classifier(".")

    print("\n✓ Model loaded successfully!")
    return model, tokenizer


def test_forward_pass(model, tokenizer):
    """Test forward pass"""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass")
    print("=" * 80)

    # Prepare input
    text = "Hello, how are"
    print(f"\nInput text: '{text}'")

    input_ids = tokenizer.encode(text, return_tensors="pt")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)

    print(f"\n✓ Forward pass successful!")
    print(f"  - Logits shape: {outputs.logits.shape}")
    print(f"  - Expected shape: (batch_size={input_ids.shape[0]}, seq_len={input_ids.shape[1]}, vocab_size={model.config.vocab_size})")

    # Get next token prediction
    next_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode([next_token_id])
    print(f"\n  - Predicted next token: '{next_token}' (ID: {next_token_id})")

    return outputs


def test_generation(model, tokenizer):
    """Test text generation"""
    print("\n" + "=" * 80)
    print("TEST 3: Text Generation")
    print("=" * 80)

    prompt = "Een aantal kleuren:"
    print(f"\nPrompt: '{prompt}'")

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Prompt token IDs: {input_ids}")

    # Generate
    print("\nGenerating text...")
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n✓ Generation successful!")
        print(f"\nGenerated text:\n{generated_text}")

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_save_and_load(model, tokenizer):
    """Test save_pretrained and from_pretrained"""
    print("\n" + "=" * 80)
    print("TEST 4: Save and Load")
    print("=" * 80)

    save_dir = "./test_olifant_model"
    print(f"\nSaving model to: {save_dir}")

    # Save
    model.save_pretrained(save_dir)
    print("✓ Model saved!")

    # Load
    print("\nLoading model from saved directory...")
    loaded_model = OlifantForCausalLM.from_pretrained(save_dir)
    loaded_model.set_tokenizer(tokenizer)

    print("✓ Model loaded from disk!")

    # Test that loaded model works
    print("\nTesting loaded model with forward pass...")
    text = "Test input"
    input_ids = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        outputs = loaded_model(input_ids)

    print(f"✓ Loaded model works! Logits shape: {outputs.logits.shape}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("OLIFANT HUGGINGFACE INTEGRATION TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Load model
        model, tokenizer = test_basic_loading()

        # Test 2: Forward pass
        test_forward_pass(model, tokenizer)

        # Test 3: Generation
        test_generation(model, tokenizer)

        # Test 4: Save and load
        test_save_and_load(model, tokenizer)

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
