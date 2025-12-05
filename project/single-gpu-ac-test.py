import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2LMHeadModel
import matplotlib.pyplot as plt

# --- GLOBAL CONFIGURATION ---
NUM_LAYERS = 12
STEPS = 10 # Number of steps for timing runs
# Use a smaller model than the FSDP one, but large enough for clear results
B_TEST = 2  # Fixed large batch size for all tests
T = 128
VOCAB_SIZE = 50000

# --- MOCK PROFILING DATA (Simulates SAC costs) ---
MOCK_MEMORY_SCORES = {
    0: 10, 1: 15, 2: 10, 3: 20,
    4: 50, 5: 10, 6: 15, 7: 55,
    8: 10, 9: 60, 10: 10, 11: 10
}
# Adaptive Parameters
ADAPTIVE_THRESHOLD_MB = 100 # Low threshold to force adaptive mode ON
SAC_BUDGET = 120 # Limited budget for the SAC tests
SAC_SCORE_THRESHOLD = 20 # Only checkpoint layers with score >= 20

# ---------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------

def _call_block(block, hidden_states):
    output = block(hidden_states)
    return output[0] if isinstance(output, tuple) else output

def run_step_with_mode(model, input_ids, mode, budget_score=None, threshold_mb=None):
    """
    Runs a single forward-backward step with a specified checkpointing mode.
    Modes: 'No-AC', 'Uniform-AC', 'SAC', 'Adaptive-AC'
    """

    # Checkpoint function that PyTorch requires for the simple implementation
    def cp_fn(block, hidden_states):
        return _call_block(block, hidden_states)

    # --- Mode Setup ---
    # Determine the checkpointing logic based on the mode
    should_checkpoint = [False] * NUM_LAYERS

    if mode == 'Uniform-AC':
        should_checkpoint = [True] * NUM_LAYERS
    elif mode == 'SAC':
        # Simple SAC: Checkpoint all layers above the score threshold
        for i in range(NUM_LAYERS):
            if MOCK_MEMORY_SCORES.get(i, 0) >= SAC_SCORE_THRESHOLD:
                should_checkpoint[i] = True
    elif mode == 'Adaptive-AC' and threshold_mb is not None and budget_score is not None:
        # Adaptive Logic (Req 1/3)
        current_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        mem_pressure = current_mem_mb > threshold_mb

        if mem_pressure:
            remaining_budget = budget_score
            # Prioritize high-score layers until budget is depleted
            sorted_scores = sorted(MOCK_MEMORY_SCORES.items(), key=lambda item: item[1], reverse=True)

            for i, score in sorted_scores:
                if score >= 15 and score <= remaining_budget:
                    should_checkpoint[i] = True
                    remaining_budget -= score


    torch.cuda.reset_peak_memory_stats()

    hidden_states = model.transformer.wte(input_ids)
    hidden_states = model.transformer.drop(hidden_states)

    # --- Run Forward Pass ---
    for i, block in enumerate(model.transformer.h):
        if should_checkpoint[i]:
            hidden_states = checkpoint(cp_fn, block, hidden_states)
        else:
            hidden_states = _call_block(block, hidden_states)

    # Final layers and backward pass
    hidden_states = model.transformer.ln_f(hidden_states)
    logits = model.lm_head(hidden_states)
    loss = logits.mean()

    try:
        loss.backward()
    except RuntimeError as e:
        if "out of memory" in str(e):
            return float('inf'), float('inf') # Indicate failure
        raise e

    peak_mem_bytes = torch.cuda.max_memory_allocated()
    return peak_mem_bytes, sum(should_checkpoint)


def benchmark_mode(model, input_ids, mode, **kwargs):
    """Runs N steps and measures average time and memory."""
    print(f"\n--- Benchmarking: {mode} ---")

    times = []

    # Warmup (Forward pass only)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        output = model(input_ids)
        _ = output.logits.mean() # Calculate loss but skip backward
    torch.cuda.empty_cache()

    # Timing and Memory Run
    peak_mem, cp_count = 0, 0
    for _ in range(STEPS):
        model.zero_grad(set_to_none=True)
        t0 = time.time()

        peak_mem, cp_count = run_step_with_mode(model, input_ids, mode, **kwargs)

        t1 = time.time()
        times.append(t1 - t0)
        torch.cuda.empty_cache()

        # Check for OOM failure
        if peak_mem == float('inf'):
             print("FATAL: Out of Memory (OOM) Error!")
             return float('inf'), float('inf'), 0

    avg_time = sum(times) / STEPS
    print(f"Time: {avg_time:.4f} sec | Peak Mem: {peak_mem / (1024**3):.2f} GB | CP Layers: {cp_count}/{NUM_LAYERS}")
    return avg_time, peak_mem, cp_count

# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = "cuda"

    # Setup Model
    cfg = GPT2Config(n_embd=256, n_layer=NUM_LAYERS, n_head=4, n_positions=T, vocab_size=VOCAB_SIZE)
    model = GPT2LMHeadModel(cfg).to(device)
    model.train()

    input_ids = torch.randint(0, VOCAB_SIZE, (B_TEST, T)).to(device)
    print(f"Benchmarking all modes with fixed Batch Size B={B_TEST} (Single GPU)")

    results = {}

    # 1. Baseline: No Activation Checkpointing (No-AC)
    t, m, c = benchmark_mode(model, input_ids, 'No-AC')
    results['No-AC'] = (t, m, c)

    # 2. Baseline: Uniform Checkpointing (Uniform-AC)
    t, m, c = benchmark_mode(model, input_ids, 'Uniform-AC')
    results['Uniform-AC'] = (t, m, c)

    # 3. Simple Selective Checkpointing (SAC)
    t, m, c = benchmark_mode(model, input_ids, 'SAC')
    results['SAC'] = (t, m, c)

    # 4. Proposed System: Adaptive Checkpointing (Adaptive-AC)
    t, m, c = benchmark_mode(model, input_ids, 'Adaptive-AC',
                            threshold_mb=ADAPTIVE_THRESHOLD_MB,
                            budget_score=SAC_BUDGET)
    results['Adaptive-AC'] = (t, m, c)

    # ---------------------------------------------------------
    # Final Comparison and Plotting
    # ---------------------------------------------------------

    labels = ["No-AC", "Uniform-AC", "SAC", "Adaptive-AC"]

    # Extracting results, filtering OOM failures
    plot_labels = [l for l in labels if results[l][0] != float('inf')]
    plot_times = [results[l][0] for l in plot_labels]
    plot_memories = [results[l][1] / (1024**3) for l in plot_labels] # Convert to GB


    # --- PLOT 1: STEP TIME ---
    plt.figure(figsize=(7, 6))
    plt.bar(plot_labels, plot_times, color=['red', 'blue', 'green', 'orange'])
    plt.ylabel("Avg Step Time (sec)")
    plt.title(f"1. Step Time Comparison (B={B_TEST}, Single GPU)")
    plt.savefig("single_gpu_time_comparison_exact_copy.png")

    # --- PLOT 2: PEAK MEMORY ---
    plt.figure(figsize=(7, 6))
    plt.bar(plot_labels, plot_memories, color=['red', 'blue', 'green', 'orange'])
    plt.ylabel("Peak Memory (GB)")
    plt.title(f"2. Peak Memory Comparison (B={B_TEST}, Single GPU)")
    plt.savefig("single_gpu_memory_comparison_exact_copy.png")

    print("\n" + "="*50)
    print("Experiment Complete. Saved two comparison plots:")
    print(" - single_gpu_time_comparison.png")
    print(" - single_gpu_memory_comparison.png")
    print("="*50)

if __name__ == "__main__":
    main()