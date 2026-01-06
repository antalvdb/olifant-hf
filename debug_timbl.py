"""
Debug script for Olifant TiMBL integration with verbose logging
"""

import torch
from transformers import AutoTokenizer
from olifant_hf import OlifantConfig, OlifantForCausalLM

def debug_forward():
    print("=" * 80)
    print("OLIFANT TIMBL DEBUG")
    print("=" * 80)
    
    # Load model locally
    config = OlifantConfig(
        vocab_size=50257,
        window_size=4,
        timbl_options="-a4 +D",
        model_prefix="edufineweb_train_000001-100k.tok.l4r0"
    )
    
    model = OlifantForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model.set_tokenizer(tokenizer)
    model._load_timbl_classifier(".")
    
    # Test input
    text = "The quick brown"
    print(f"\n[INPUT] Text: '{text}'")
    
    input_ids = tokenizer.encode(text, return_tensors="pt")
    print(f"[INPUT] Token IDs: {input_ids.tolist()}")
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    print(f"[INPUT] Tokens: {tokens}")
    
    print("\n" + "-" * 80)
    print("PROCESSING EACH POSITION")
    print("-" * 80)
    
    batch_idx = 0
    sequence_ids = input_ids[batch_idx]
    all_tokens = model._ids_to_tokens(sequence_ids)
    print(f"\n[TOKENS] All tokens from _ids_to_tokens: {all_tokens}")
    
    for pos in range(len(all_tokens)):
        print(f"\n{'='*40}")
        print(f"POSITION {pos}")
        print(f"{'='*40}")
        
        # Get context window (including current token to predict NEXT token)
        context_tokens = all_tokens[max(0, pos + 1 - config.window_size):pos + 1]
        print(f"[CONTEXT] Raw context (before padding): {context_tokens}")
        
        # Pad context
        padded_context = model._pad_prompt(context_tokens, config.window_size)
        print(f"[CONTEXT] Padded context (window_size={config.window_size}): {padded_context}")
        
        # Query TiMBL
        try:
            classlabel, distribution, distance = model.timbl_classifier.classify(padded_context)
            print(f"[TIMBL] Class label: '{classlabel}'")
            print(f"[TIMBL] Distance: {distance}")
            print(f"[TIMBL] Distribution type: {type(distribution)}")
            
            if isinstance(distribution, dict):
                print(f"[TIMBL] Distribution (top 10):")
                sorted_dist = sorted(distribution.items(), key=lambda x: -x[1])[:10]
                for token, score in sorted_dist:
                    token_id = model._token_to_id(token)
                    print(f"         '{token}' (id={token_id}): {score:.4f}")
            else:
                print(f"[TIMBL] Distribution (raw): {distribution[:200] if len(str(distribution)) > 200 else distribution}")
            
            # Convert to logits
            logits = model._distribution_to_logits(distribution, distance)
            top_logits, top_indices = torch.topk(logits, 5)
            print(f"[LOGITS] Top 5 predictions:")
            for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
                token = tokenizer.decode([idx.item()])
                print(f"         {i+1}. '{token}' (id={idx.item()}): logit={logit.item():.4f}")
                
        except Exception as e:
            print(f"[ERROR] Classification failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("GENERATION TEST")
    print("=" * 80)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print(f"\n[GENERATE] Input IDs:  {input_ids.tolist()}")
    print(f"[GENERATE] Output IDs: {output_ids.tolist()}")
    print(f"[GENERATE] Generated text: '{tokenizer.decode(output_ids[0])}'")
    
    # Show each generated token
    print(f"\n[GENERATE] Token-by-token:")
    for i, token_id in enumerate(output_ids[0]):
        token = tokenizer.decode([token_id.item()])
        marker = " <-- generated" if i >= input_ids.shape[1] else ""
        print(f"         [{i}] id={token_id.item():5d} -> '{token}'{marker}")

if __name__ == "__main__":
    debug_forward()
