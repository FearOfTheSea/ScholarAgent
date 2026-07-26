"""Pre-trains a miniature custom GPT model on a small sample corpus to demonstrate Chapters 2 and 5 concepts."""

import math
from pathlib import Path
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from scholar_agent.infrastructure.adapters.scratch_gpt.gpt_model import GPTModel

# 1. Prepare a tiny sample corpus
CORPUS = """
ScholarAgent is an educational study assistant that helps students.
It is built with clean architecture, hexagonal boundaries, and ports and adapters.
Clean architecture prioritizes readability, maintainability, and testability over features.
LangChain and LangGraph are isolated behind output ports in the infrastructure layer.
This ensures the domain and application layers remain independent of external frameworks.
Local models run completely offline on your laptop using Ollama and FAISS repositories.
To study effectively, use study tools like flashcards, quizzes, and summaries.
Students love clean software engineering design because it is elegant and simple.
"""


def main() -> None:
    print("=== Build-a-LLM-from-Scratch: Mini Pre-training Demo ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device}\n")

    # 2. Tokenize and prepare data (Chapter 2 & 5 concepts)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    token_ids = tokenizer.encode(CORPUS)
    print(f"Dataset tokenized: {len(token_ids)} tokens found.")

    context_length = 16
    x_data = []
    y_data = []

    # Create sliding window inputs and targets
    for i in range(len(token_ids) - context_length):
        x_data.append(token_ids[i : i + context_length])
        y_data.append(token_ids[i + 1 : i + context_length + 1])

    X = torch.tensor(x_data, dtype=torch.long, device=device)
    Y = torch.tensor(y_data, dtype=torch.long, device=device)
    print(f"Generated {len(X)} training sequences of length {context_length}.")

    # 3. Instantiate miniature custom GPT model (Chapter 4 concepts)
    mini_config = {
        "vocab_size": 50257,
        "context_length": context_length,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 4,
        "bias": True,
    }
    
    model = GPTModel(mini_config).to(device)
    print(f"Initialized miniature GPTModel: {sum(p.numel() for p in model.parameters()):,} parameters.")

    # 4. Generate text before training to show baseline (Chapter 5)
    prompt = "ScholarAgent is"
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    print("\n--- Text Generation BEFORE training ---")
    print(f"Prompt: '{prompt}'")
    with torch.no_grad():
        generated_before = generate_text(model, prompt_ids, max_new_tokens=15, context_len=context_length)
    print(f"Output: '{tokenizer.decode(generated_before[0].tolist())}'")

    # 5. Training Loop (Chapter 5 concepts)
    epochs = 40
    batch_size = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n--- Starting training for {epochs} epochs (batch_size={batch_size}) ---")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = math.ceil(len(X) / batch_size)
        
        # Shuffle inputs each epoch
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        for i in range(0, len(X), batch_size):
            x_batch = X_shuffled[i : i + batch_size]
            y_batch = Y_shuffled[i : i + batch_size]

            optimizer.zero_grad()
            logits = model(x_batch)
            
            # Reshape logits to (batch_size * sequence_length, vocab_size)
            # Reshape targets to (batch_size * sequence_length)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Average Loss: {avg_loss:.4f}")

    # 6. Generate text after training (Chapter 5)
    model.eval()
    print("\n--- Text Generation AFTER training ---")
    print(f"Prompt: '{prompt}'")
    with torch.no_grad():
        generated_after = generate_text(model, prompt_ids, max_new_tokens=15, context_len=context_length)
    print(f"Output: '{tokenizer.decode(generated_after[0].tolist())}'")
    checkpoint_path = Path("data/mini_gpt.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved MiniGPT checkpoint to: {checkpoint_path}")
    print("\nObserve how the model learns vocabulary and simple transitions from the training text!")


def generate_text(model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_len: int) -> torch.Tensor:
    """Helper to autoregressively generate tokens from a prompt. Call within torch.no_grad()."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_len:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=-1)
    return idx


if __name__ == "__main__":
    main()
