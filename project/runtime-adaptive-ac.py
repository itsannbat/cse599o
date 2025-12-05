import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2LMHeadModel
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Utility: Helper for block execution
# ---------------------------------------------------------------------

def _call_block(block, hidden_states):
    output = block(hidden_states)
    return output[0] if isinstance(output, tuple) else output

# ---------------------------------------------------------------------
# 1. RUNTIME ADAPTIVE LOGIC (The New Feature)
# ---------------------------------------------------------------------

def run_adaptive_step(model, input_ids, memory_threshold_mb):
    """
    Decides whether to checkpoint dynamically based on current memory usage.
    """
    # Convert MB to Bytes
    threshold_bytes = memory_threshold_mb * 1024 * 1024

    # 1. Embeddings
    hidden_states = model.transformer.wte(input_ids)
    hidden_states = model.transformer.drop(hidden_states)

    decisions = [] # To track what the model decided to do

    # 2. Iterate over blocks with DYNAMIC decision making
    for i, block in enumerate(model.transformer.h):

        # [CRITICAL STEP] Measure current memory
        current_mem = torch.cuda.memory_allocated()

        # PREDICTION/HEURISTIC:
        # If we are above the threshold, we must checkpoint to save space.
        # If we are below, we skip checkpointing to save time.
        if current_mem > threshold_bytes:
            hidden_states = checkpoint(_call_block, block, hidden_states)
            decisions.append(1) # 1 = Checkpointed
        else:
            hidden_states = _call_block(block, hidden_states)
            decisions.append(0) # 0 = Kept in memory (No CP)

    # 3. Final layers
    hidden_states = model.transformer.ln_f(hidden_states)
    logits = model.lm_head(hidden_states)

    loss = logits.mean()
    loss.backward()

    return loss.item(), decisions

# ---------------------------------------------------------------------
# Benchmark Wrapper
# ---------------------------------------------------------------------

def benchmark_adaptive(model, input_ids, threshold_mb):
    torch.cuda.empty_cache()
    model.zero_grad(set_to_none=True)

    # Warmup
    print(f"--- Warming up for Threshold {threshold_mb}MB ---")
    loss, decisions = run_adaptive_step(model, input_ids, threshold_mb)

    # Analyze what happened
    cp_count = sum(decisions)
    print(f"Decisions (1=CP, 0=NoCP): {decisions}")
    print(f"Layers Checkpointed: {cp_count}/{len(decisions)}")

    # Timing run
    start_time = time.time()
    for _ in range(5): # Run 5 steps
        model.zero_grad(set_to_none=True)
        run_adaptive_step(model, input_ids, threshold_mb)
    end_time = time.time()

    avg_time = (end_time - start_time) / 5
    return avg_time, cp_count

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Adaptive memory checkpointing requires a GPU.")
        return

    device = "cuda"
    print(f"Device: {device}")

    # Get total GPU memory to give context
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    print(f"Total GPU Memory: {total_mem:.0f} MB")

    # Setup Model (Same as before)
    cfg = GPT2Config(n_embd=512, n_layer=12, n_head=8, n_positions=512, vocab_size=50000)
    model = GPT2LMHeadModel(cfg).to(device)
    model.train()

    # Input Data
    B, T = 8, 256
    input_ids = torch.randint(0, 50000, (B, T)).to(device)

    # ---------------------------------------------------------
    # EXPERIMENT: Vary the threshold to see adaptation
    # ---------------------------------------------------------

    # Scenario A: High Threshold (Acts like No-Checkpointing)
    # We set the limit very high, so it never thinks it's full.
    time_high, count_high = benchmark_adaptive(model, input_ids, threshold_mb=total_mem * 0.9)

    # Scenario B: Low Threshold (Acts like Uniform Checkpointing)
    # We set the limit very low, so it panics and checkpoints everything.
    time_low, count_low = benchmark_adaptive(model, input_ids, threshold_mb=100)

    # Scenario C: "Sweet Spot" (The Adaptive Behavior)
    # Set this to somewhere in the middle where it switches halfway through.
    # You might need to tweak this number based on your specific GPU!
    # Try finding a value where 'decisions' looks like [0, 0, 0, 1, 1, 1...]
    target_mb = 1000 # <-- TWEAK THIS NUMBER
    time_adapt, count_adapt = benchmark_adaptive(model, input_ids, threshold_mb=target_mb)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    labels = [f"High Limit\n(CP={count_high})", f"Low Limit\n(CP={count_low})", f"Adaptive\n(CP={count_adapt})"]
    times = [time_high, time_low, time_adapt]

    plt.figure(figsize=(8,5))
    bars = plt.bar(labels, times, color=['green', 'red', 'blue'])
    plt.ylabel("Time per step (sec)")
    plt.title("Adaptive Activation Checkpointing Runtime")

    # Add labels on bars
    for bar, t in zip(bars, times):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{t:.4f}s", ha='center', va='bottom')

    plt.savefig("adaptive_results.png")
    print("\nSaved adaptive_results.png")

if __name__ == "__main__":
    main()