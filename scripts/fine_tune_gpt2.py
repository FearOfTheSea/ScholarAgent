"""Demonstrate instruction fine-tuning with the custom GPT and GPT-2 weights."""

from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from scholar_agent.infrastructure.adapters.scratch_gpt.gpt_model import (
    GPT2_124M_CONFIG,
    GPTModel,
    load_gpt2_weights,
)


class InstructionSample(TypedDict):
    """One supervised instruction and expected response."""

    instruction: str
    response: str


# 1. Prepare a small QA instruction dataset (Chapter 6)
INSTRUCTION_DATA: tuple[InstructionSample, ...] = (
    {
        "instruction": "What is the primary architectural rule of ScholarAgent?",
        "response": (
            "Dependencies must always point inward from presentation towards "
            "the domain layer."
        ),
    },
    {
        "instruction": "Which layers are forbidden from importing frameworks?",
        "response": (
            "The domain and application layers must never depend on external "
            "frameworks such as LangChain."
        ),
    },
    {
        "instruction": "What is the role of an adapter in Ports and Adapters?",
        "response": (
            "Adapters implement output ports that connect the application core "
            "to external services."
        ),
    },
    {
        "instruction": "Where are FAISS and PyMuPDF located in the project?",
        "response": (
            "They are infrastructure adapters that implement application output ports."
        ),
    },
)


def format_instruction(sample: InstructionSample) -> str:
    """Format QA pair according to instruction prompt template."""
    return (
        f"Instruction: {sample['instruction']}\n"
        f"Response: {sample['response']}<|endoftext|>"
    )


def main() -> None:
    print("=== Build-a-LLM-from-Scratch: Instruction Fine-Tuning Demo ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device}\n")

    # 2. Load tokenizer and pre-trained weights (Chapter 5)
    print("Loading pre-trained GPT-2 (124M) tokenizer and weights...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_hf = GPT2LMHeadModel.from_pretrained("gpt2")

    # 3. Instantiate custom model and load weights
    model = GPTModel(GPT2_124M_CONFIG)
    load_gpt2_weights(model, gpt_hf)
    model.to(device)
    print("Pre-trained weights successfully loaded into custom GPTModel structure.\n")

    # 4. Prepare data sequences
    formatted_texts = [format_instruction(sample) for sample in INSTRUCTION_DATA]
    encoded_sequences = [tokenizer.encode(text) for text in formatted_texts]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("GPT-2 tokenizer does not define an EOS token.")

    # Pad sequences to match the longest one for batching
    max_len = max(len(seq) for seq in encoded_sequences)
    padded_sequences = []
    target_sequences = []

    for seq in encoded_sequences:
        # Pad with EOS token id (50256)
        padded = seq + [eos_token_id] * (max_len - len(seq))
        # Target shifts left by 1
        padded_sequences.append(padded)
        target_sequences.append(padded[1:] + [eos_token_id])

    inputs = torch.tensor(padded_sequences, dtype=torch.long, device=device)
    targets = torch.tensor(target_sequences, dtype=torch.long, device=device)

    # 5. Run a miniature Fine-Tuning loop (Chapter 6 concepts)
    # Using small learning rate to perform parameter-efficient/fine-tuning updates
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=eos_token_id)

    epochs = 5
    print(
        "Starting instruction fine-tuning for "
        f"{epochs} epochs over {len(INSTRUCTION_DATA)} samples..."
    )
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(inputs)

        # Calculate loss only on the generated responses (ignore padding)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}/{epochs} | Fine-Tuning Loss: {loss.item():.4f}")

    # 6. Save checkpoint
    checkpoint_path = Path("data/scratch_gpt_finetuned.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # 7. Test fine-tuned inference
    model.eval()
    test_prompt = (
        "Instruction: What is the primary architectural rule of ScholarAgent?\n"
        "Response:"
    )
    test_ids = torch.tensor(
        tokenizer.encode(test_prompt), dtype=torch.long, device=device
    ).unsqueeze(0)

    print("\n--- Testing Custom Fine-Tuned Model Inference ---")
    print(f"Prompt: '{test_prompt}'")

    generated = test_ids
    context_len = GPT2_124M_CONFIG["context_length"]

    # Generate tokens autoregressively
    for _ in range(20):
        idx_cond = generated[:, -context_len:]
        with torch.no_grad():
            logits = model(idx_cond)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == eos_token_id:
            break
        generated = torch.cat((generated, next_token), dim=-1)

    completion_ids = generated[0, len(test_ids[0]) :].tolist()
    completion_text = tokenizer.decode(completion_ids)
    print(f"Generated response: '{completion_text.strip()}'")


if __name__ == "__main__":
    main()
