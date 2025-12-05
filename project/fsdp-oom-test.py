import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2LMHeadModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

# --- GLOBAL CONFIGURATION (Same Large Model) ---
N_LAYERS = 24
SEQ_LEN = 1024
VOCAB_SIZE = 50257
STEPS = 2 # Only run 2 steps per batch size attempt

# Mock Scores and FSDP/Dist Setup functions remain the same as in fsdp_train.py
# (Ensure you copy them over or reuse the file contents!)

# FSDP SETUP UTILITIES (Copy from previous code)
def init_dist(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def get_fsdp_wrap_policy(module):
    return partial(transformer_auto_wrap_policy, transformer_layer_cls={module.transformer.h[0].__class__})

# ADAPTIVE FSDP-SAC STEP (Copy run_fsdp_sac_step function, or use the single argument version below)

# --- Simplified run_step_with_mode for OOM testing ---
def run_step_with_mode(model, input_ids, mode, memory_threshold_mb=None, budget_score=None):
    """Simplified run_step that returns True on success and False on OOM."""

    # Logic to set checkpointing decisions based on 'mode' goes here (Adaptive, Uniform, None)
    # The full logic from the previous fsdp_train.py's run_fsdp_sac_step function must be adapted

    # IMPORTANT: For this OOM test, we simplify the FSDP-SAC check to ensure it's ON for 'Adaptive' mode.
    # We will use the 'Uniform' logic for the Uniform test, and the 'None' logic for the No-CP test.

    # ... (Your previous checkpointing logic from final-experiment.py, adapted for distributed FSDP) ...

    # --- Simplified checkpointing based on mode ---

    decisions = []

    hidden_states = model.module.transformer.wte(input_ids)
    hidden_states = model.module.transformer.drop(hidden_states)

    for i, block in enumerate(model.module.transformer.h):
        use_cp = False
        if mode == 'Uniform':
            use_cp = True
        elif mode == 'Adaptive' and MOCK_MEMORY_SCORES[i] > 25: # Simple SAC trigger for this test
            use_cp = True

        if use_cp:
            hidden_states = checkpoint(_call_block, block, hidden_states)
        else:
            hidden_states = _call_block(block, hidden_states)
        decisions.append(1 if use_cp else 0)

    # Final layers and backward pass
    hidden_states = model.module.transformer.ln_f(hidden_states)
    logits = model.module.lm_head(hidden_states)
    loss = logits.mean()
    loss = loss / dist.get_world_size()

    try:
        model.zero_grad(set_to_none=True)
        # Force a proper forward/backward pass
        loss.backward()
        dist.barrier()
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False
        raise e

# ---------------------------------------------------------------------
# OOM SEARCH FUNCTION (Req 4 demonstrated by finding max B)
# ---------------------------------------------------------------------

def find_max_batch_size(fsdp_model, mode, max_B=128, initial_B=8):
    """Searches for the maximum safe global batch size B."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    max_safe_B = 0

    # Start the search from the initial safe batch size
    B = initial_B

    while B <= max_B:
        local_B = B // world_size

        if rank == 0:
            print(f"\n[Mode: {mode}] Testing Global B={B} (Local B={local_B})...")

        # Prepare inputs for the current B
        input_ids = torch.randint(0, VOCAB_SIZE, (local_B, SEQ_LEN)).to(rank)

        is_safe = True
        # Run multiple steps to ensure OOM is not transient
        for _ in range(STEPS):
            if not run_step_with_mode(fsdp_model, input_ids, mode=mode):
                is_safe = False
                break

        # All ranks synchronize the safety status
        safety_status = torch.tensor(1 if is_safe else 0, device=rank)
        # All-reduce to check if *any* rank failed (if safety_status_sum < world_size)
        dist.all_reduce(safety_status, op=dist.ReduceOp.SUM)

        if safety_status.item() < world_size:
            if rank == 0:
                print(f"[Mode: {mode}] OOM detected at B={B}. Max safe B found: {max_safe_B}")
            return max_safe_B

        if rank == 0:
            max_safe_B = B

        # Increment B (we can use a power-of-two or linear step, using linear for safety)
        B += 8

    return max_safe_B

# ---------------------------------------------------------------------
# MAIN DISTRIBUTED FUNCTION
# ---------------------------------------------------------------------

def main_worker():
    # Setup distributed environment
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    init_dist(rank, world_size)

    # Configure model
    cfg = GPT2Config(n_embd=1024, n_layer=N_LAYERS, n_head=16,
                     n_positions=SEQ_LEN, vocab_size=VOCAB_SIZE)
    model = GPT2LMHeadModel(cfg)

    # FSDP setup
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=get_fsdp_wrap_policy(model),
        device_id=rank
    )
    fsdp_model.train()

    # Ensure all ranks have the same max_safe_B (optional, but clean)
    final_results = {}

    # 1. No Checkpointing Test
    max_B_none = find_max_batch_size(fsdp_model, mode='None')
    final_results['No-CP'] = max_B_none
    
    # 2. Uniform Checkpointing Test
    max_B_uniform = find_max_batch_size(fsdp_model, mode='Uniform')
    final_results['Uniform-CP'] = max_B_uniform

    # 3. Adaptive FSDP-SAC Test
    max_B_adaptive = find_max_batch_size(fsdp_model, mode='Adaptive')
    final_results['Adaptive-SAC'] = max_B_adaptive

    dist.barrier()

    if rank == 0:
        print("\n" + "="*50)
        print("✅ FINAL OOM STRESS TEST RESULTS (Max Safe Global Batch Size)")
        for mode, max_b in final_results.items():
            print(f"[{mode}]: B_max = {max_b}")
        print("="*50)

    dist.destroy_process_group()

if __name__ == "__main__":
    main_worker()