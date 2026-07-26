"""Demonstrates Chapter 6 Instruction Fine-Tuning concepts using the custom GPTModel architecture and GPT-2 weights."""

import os
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scholar_agent.infrastructure.adapters.scratch_gpt.gpt_model import GPTModel, load_gpt2_weights
from scholar_agent.infrastructure.adapters.scratch_gpt.scratch_gpt_adapter import GPT_CONFIG_124M

# 1. Prepare a small QA instruction dataset (Chapter 6)
INSTRUCTION_DATA = [
    {
        "instruction": "What is the primary architectural rule of ScholarAgent?",
        "response": "Dependencies must always point inward from presentation towards the domain layer."
    },
    {
        "instruction": "Which layers are forbidden from importing frameworks?",
        "response": "The domain and application layers must never depend on any external frameworks like LangChain."
    },
    {
        "instruction": "What is the role of an adapter in Ports and Adapters?",
        "response": "Adapters implement the output ports to connect the application core to external services."
    },
    {
        "instruction": "Where are FAISS and PyMuPDF located in the project?",
        "response": "They are infrastructure adapters that implement defined application output ports."
    }
]


def format_instruction(sample: dict) -> str:
    """Format QA pair according to instruction prompt template."""
    return f"Instruction: {sample['instruction']}\nResponse: {sample['response']}<|endoftext|>"


def main() -> None:
    print("=== Build-a-LLM-from-Scratch: Instruction Fine-Tuning Demo ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device}\n")

    # 2. Load tokenizer and pre-trained weights (Chapter 5)
    print("Loading pre-trained GPT-2 (124M) tokenizer and weights...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # 3. Instantiate custom model and load weights
    model = GPTModel(GPT_CONFIG_124M)
    load_gpt2_weights(model, gpt_hf)
    model.to(device)
    print("Pre-trained weights successfully loaded into custom GPTModel structure.\n")

    # 4. Prepare data sequences
    formatted_texts = [format_instruction(sample) for sample in INSTRUCTION_DATA]
    encoded_sequences = [tokenizer.encode(text) for text in formatted_texts]
    
    # Pad sequences to match the longest one for batching
    max_len = max(len(seq) for seq in encoded_sequences)
    padded_sequences = []
    target_sequences = []
    
    for seq in encoded_sequences:
        # Pad with EOS token id (50256)
        padded = seq + [tokenizer.eos_token_id] * (max_len - len(seq))
        # Target shifts left by 1
        padded_sequences.append(padded)
        target_sequences.append(padded[1:] + [tokenizer.eos_token_id])

    X = torch.tensor(padded_sequences, dtype=torch.long, device=device)
    Y = torch.tensor(target_sequences, dtype=torch.long, device=device)

    # 5. Run a miniature Fine-Tuning loop (Chapter 6 concepts)
    # Using small learning rate to perform parameter-efficient/fine-tuning updates
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id) # ignore padding loss
    
    epochs = 5
    print(f"Starting instruction fine-tuning for {epochs} epochs over {len(INSTRUCTION_DATA)} samples...")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X)
        
        # Calculate loss only on the generated responses (ignore padding)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}/{epochs} | Fine-Tuning Loss: {loss.item():.4f}")

    # 6. Save checkpoint
    os.makedirs("data", exist_ok=True)
    checkpoint_path = "data/scratch_gpt_finetuned.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # 7. Test fine-tuned inference
    model.eval()
    test_prompt = "Instruction: What is the primary architectural rule of ScholarAgent?\nResponse:"
    test_ids = torch.tensor(tokenizer.encode(test_prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    print("\n--- Testing Custom Fine-Tuned Model Inference ---")
    print(f"Prompt: '{test_prompt}'")
    
    generated = test_ids
    context_len = GPT_CONFIG_124M["context_length"]
    
    # Generate tokens autoregressively
    for _ in range(20):
        idx_cond = generated[:, -context_len:]
        with torch.no_grad():
            logits = model(idx_cond)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated = torch.cat((generated, next_token), dim=-1)

    completion_ids = generated[0, len(test_ids[0]):].tolist()
    completion_text = tokenizer.decode(completion_ids)
    print(f"Generated response: '{completion_text.strip()}'")


if __name__ == "__main__":
    main()
