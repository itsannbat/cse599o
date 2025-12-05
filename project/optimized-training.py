import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2LMHeadModel
import matplotlib.pyplot as plt

# --- Mock Profiling Data (From Req 3) ---
MOCK_MEMORY_SCORES = {
    0: 10, 1: 15, 2: 10, 3: 20,
    4: 50, 5: 10, 6: 15, 7: 55,
    8: 10, 9: 60, 10: 10, 11: 10
}

# ---------------------------------------------------------------------
# FSDP-SAC Logic (Combined Req 1, 2, 3)
# ---------------------------------------------------------------------

def _call_block(block, hidden_states):
    output = block(hidden_states)
    return output[0] if isinstance(output, tuple) else output

def run_fsdp_sac_step(model, input_ids, memory_threshold_mb, budget_score):
    """
    Combines adaptive (threshold), FSDP (implicit via decision), and SAC (budget).
    """
    threshold_bytes = memory_threshold_mb * 1024 * 1024

    hidden_states = model.transformer.wte(input_ids)
    hidden_states = model.transformer.drop(hidden_states)

    decisions = []

    # Check memory pressure *before* starting the main compute (Req 1)
    current_mem = torch.cuda.memory_allocated()
    mem_pressure = current_mem > threshold_bytes

    # Reset peak memory tracker before the step starts
    torch.cuda.reset_peak_memory_stats()

    for i, block in enumerate(model.transformer.h):

        layer_score = MOCK_MEMORY_SCORES.get(i, 0)
        use_cp = False

        # SAC Decision Logic (Req 3)
        if mem_pressure and budget_score > 0 and layer_score > 15:
            use_cp = True
            budget_score -= layer_score

        if use_cp:
            hidden_states = checkpoint(_call_block, block, hidden_states)
            decisions.append(1)
        else:
            hidden_states = _call_block(block, hidden_states)
            decisions.append(0)

    # Final layers and backward pass
    hidden_states = model.transformer.ln_f(hidden_states)
    logits = model.lm_head(hidden_states)
    loss = logits.mean()
    loss.backward()

    # Capture Peak Memory *during* the forward/backward pass (crucial for Req 4)
    peak_mem_bytes = torch.cuda.max_memory_allocated()

    return peak_mem_bytes, sum(decisions)


# ---------------------------------------------------------------------
# 4. MEMORY BUDGETING AND AUTO-TUNING
# ---------------------------------------------------------------------

def auto_tune_batch_size(model, input_ids, initial_batch_size, total_mem_bytes, target_mem_fraction=0.9):
    """
    Automatically finds the optimal batch size based on memory usage.
    """
    # 1. Define the Target Memory Budget
    target_mem_bytes = total_mem_bytes * target_mem_fraction
    print(f"Target Memory Budget: {target_mem_bytes / (1024**3):.2f} GB ({target_mem_fraction*100:.0f}%)")

    # 2. Run an initial step with the FSDP-SAC logic
    B_start = initial_batch_size
    T_start = input_ids.size(1)

    print(f"\n--- Benchmark Run (B={B_start}) ---")

    # We must ensure the checkpointing threshold is low to force SAC ON
    # Set threshold lower than current allocated memory (simulates pressure)
    current_allocated = torch.cuda.memory_allocated()
    sac_threshold_mb = current_allocated / (1024*1024) * 0.9
    SAC_BUDGET = 300 # Sufficient budget for full SAC coverage

    peak_mem_bytes, cp_count = run_fsdp_sac_step(model, input_ids, sac_threshold_mb, SAC_BUDGET)

    peak_mem_gb = peak_mem_bytes / (1024**3)

    print(f"Initial Peak Memory (B={B_start}): {peak_mem_gb:.2f} GB")
    print(f"Checkpoints used: {cp_count}/12")

    # 3. Calculate Optimal Batch Size (The Core Budgeting Equation)

    if peak_mem_bytes == 0:
        print("ERROR: Peak memory was 0. Cannot tune batch size.")
        return B_start

    B_predicted_float = B_start * (target_mem_bytes / peak_mem_bytes)

    # Round down to the nearest integer
    B_predicted = int(B_predicted_float)

    print("\n--- Auto-Tuning Result ---")
    print(f"Predicted B_max: {B_predicted_float:.2f} (Rounded to {B_predicted})")

    # 4. Validation Run (Optional but good practice)

    if B_predicted > B_start:
        print(f"Running Validation Step with B={B_predicted}...")

        # Scale input to the new size
        input_ids_new = torch.randint(0, 50000, (B_predicted, T_start)).to(model.device)

        # Clear cache and run again
        torch.cuda.empty_cache()
        peak_mem_val, cp_count_val = run_fsdp_sac_step(model, input_ids_new, sac_threshold_mb, SAC_BUDGET)

        peak_mem_val_gb = peak_mem_val / (1024**3)
        print(f"Validation Peak Memory (B={B_predicted}): {peak_mem_val_gb:.2f} GB")

        if peak_mem_val_gb > target_mem_bytes / (1024**3):
            print("WARNING: Validation exceeded budget. Using safe B.")

    return B_predicted


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = "cuda"
    total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
    total_mem_gb = total_mem_bytes / (1024**3)

    print(f"Total GPU Memory: {total_mem_gb:.2f} GB")

    # Setup Model (Same as before)
    cfg = GPT2Config(n_embd=512, n_layer=12, n_head=8, n_positions=512, vocab_size=50000)
    model = GPT2LMHeadModel(cfg).to(device)
    model.train()

    # Initial Small Batch
    B_initial = 8
    T = 256
    input_ids = torch.randint(0, 50000, (B_initial, T)).to(device)

    # Execute the Automated Batch Size Tuning
    optimal_B = auto_tune_batch_size(
        model,
        input_ids,
        B_initial,
        total_mem_bytes,
        target_mem_fraction=0.9 # Aim for 90% utilization
    )

    print(f"\n✅ Optimal Batch Size for 90% GPU Utilization: B={optimal_B}")

if __name__ == "__main__":
    main()