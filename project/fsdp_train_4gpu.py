import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import GPT2Config, GPT2LMHeadModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
from functools import partial

# --- GLOBAL CONFIGURATION ---
# SCALED PARAMETERS
N_LAYERS = 32           # INCREASED LAYERS
EMBED_SIZE = 1280       # INCREASED EMBEDDING SIZE
OPTIMAL_BATCH_SIZE = 12 # INCREASED GLOBAL BATCH SIZE (Local B=4)
SEQ_LEN = 1024
VOCAB_SIZE = 50257
STEPS = 5
NUM_GPUS = 4            # Target number of GPUs

# --- MOCK PROFILING DATA (Simulates SAC costs, expanded to 32 layers) ---
# Layers 12, 20, 28 are simulated as the most memory-expensive
MOCK_MEMORY_SCORES = {
    i: 10 + (40 if i % 8 == 4 else 0) + (50 if i in [12, 20, 28] else 0)
    for i in range(N_LAYERS)
}
SAC_SCORE_THRESHOLD = 50

# ---------------------------------------------------------------------
# FSDP SETUP UTILITIES
# ---------------------------------------------------------------------

def init_dist(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def get_fsdp_wrap_policy(module):
    layer_cls = module.transformer.h[0].__class__
    return partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})

# ---------------------------------------------------------------------
# DECLARATIVE SAC INTEGRATION (Req 2 & 3)
# ---------------------------------------------------------------------

def sac_checkpoint_policy(module: torch.nn.Module, sac_decisions: list[bool]) -> bool:
    # This logic checks if the FSDP-wrapped module corresponds to one of our
    # pre-determined high-cost layers.
    module_name = module._get_name()

    for i, decision in enumerate(sac_decisions):
        if f'.h.{i}' in module_name and decision:
            return True
        if hasattr(module, 'layer_idx') and module.layer_idx == i and decision:
             return True

    return False

def get_sac_decisions(n_layers: int, score_threshold: int) -> list[bool]:
    """Determines which layers should be checkpointed based on the MOCK scores."""
    decisions = []
    for i in range(n_layers):
        score = MOCK_MEMORY_SCORES.get(i, 0)
        decisions.append(score >= score_threshold)
    return decisions

# ---------------------------------------------------------------------
# MAIN DISTRIBUTED FUNCTION
# ---------------------------------------------------------------------

def main_worker():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    init_dist(rank, world_size)

    local_batch_size = OPTIMAL_BATCH_SIZE // world_size

    # 1. Setup Model (LARGER CONFIG)
    cfg = GPT2Config(n_embd=EMBED_SIZE, n_layer=N_LAYERS, n_head=16,
                     n_positions=SEQ_LEN, vocab_size=VOCAB_SIZE, **use_cache=False**) # <-- ADD THIS
    model = GPT2LMHeadModel(cfg)

    # ... Inside main_worker ...

    # 2. Adaptive SAC Decision (Req 3)
    sac_decisions = get_sac_decisions(N_LAYERS, SAC_SCORE_THRESHOLD)
    num_cp_layers = sum(sac_decisions)

    if rank == 0:
        print(f"SAC Decision: Checkpointing {num_cp_layers}/{N_LAYERS} Layers.")

    # 3. Apply Checkpointing Wrapper (The MOST Backwards-Compatible Fix)

    new_h_blocks = nn.ModuleList()

    # Manually iterate and wrap only the layers selected by SAC
    for i, block in enumerate(model.transformer.h):

        # Check if the SAC logic dictates checkpointing this layer
        if sac_decisions[i]:
            # Wrap the block with the checkpoint wrapper
            # This is the oldest, most reliable way to pre-wrap modules for FSDP
            wrapped_block = checkpoint_wrapper(
                block,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
            new_h_blocks.append(wrapped_block)
        else:
            # Keep the original block
            new_h_blocks.append(block)

    # Replace the model's transformer blocks with the list containing wrapped modules
    model.transformer.h = new_h_blocks

    # 4. FSDP Wrap
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=get_fsdp_wrap_policy(model),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank
    )

    # ... rest of the code remains the same ...

    input_ids = torch.randint(0, VOCAB_SIZE, (local_batch_size, SEQ_LEN)).to(rank)

    if rank == 0:
        print(f"--- Distributed FSDP-SAC Setup (4-GPU) ---")
        print(f"Model: {N_LAYERS}L, {EMBED_SIZE}E. Global B: {OPTIMAL_BATCH_SIZE}")
        print(f"SAC Decision: Checkpointing {num_cp_layers}/{N_LAYERS} Layers.")

    # Warmup Step
    fsdp_model.train()
    fsdp_model.zero_grad(set_to_none=True)
    with torch.no_grad():
        output = fsdp_model(input_ids,use_cache=False)
        _ = output.logits.mean()

    # Timing and Memory Run
    times = []

    for step in range(STEPS):
        fsdp_model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.time()

        output = fsdp_model(input_ids, use_cache=False)
        loss = output.logits.mean()

        loss = loss / world_size
        loss.backward()

        torch.cuda.synchronize()
        t1 = time.time()

        peak_mem = torch.cuda.max_memory_allocated()
        times.append(t1 - t0)

        dist.barrier()

        if rank == 0:
            print(f"[Rank 0] Step {step+1}: Time={t1-t0:.4f}s | Peak Mem={peak_mem/(1024**3):.2f} GB")

    avg_time = sum(times) / STEPS

    if rank == 0:
        print(f"\n✅ Final Average Step Time (B={OPTIMAL_BATCH_SIZE}): {avg_time:.4f} seconds")

    dist.destroy_process_group()

if __name__ == "__main__":
    main_worker()