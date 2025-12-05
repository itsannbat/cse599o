import torch
import torch.nn as nn
import torch.nn.init as init
import time
import datetime
import math
import random
import os

from torch.distributed import init_process_group, destroy_process_group, distributed_c10d as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
# FIX: Import the transformer auto wrap policy helper from the correct location
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl
)
from functools import partial

# --- GLOBAL CONFIGURATION (ADJUST THESE FOR EXPERIMENTS) ---

# SCALED MODEL PARAMETERS (Adjusted to prevent OOM on 24GB GPUs)
N_LAYERS = 24           # Total transformer layers (was 32)
EMBED_SIZE = 1024       # Embedding dimension (was 1280)
SEQ_LEN = 1024          # Sequence length
VOCAB_SIZE = 50257      # Mock vocabulary size
STEPS = 5               # Number of steps to benchmark (after warmup)
NUM_GPUS = 4            # Number of GPUs used
OPTIMAL_BATCH_SIZE = 12 # Global Batch Size (Local Batch Size is OPTIMAL_BATCH_SIZE / NUM_GPUS = 3)

# DISTRIBUTED SETUP FIX
# FIX: Changed port to reduce chance of conflict
MASTER_PORT = '29502'
COMMUNICATION_TIMEOUT_S = 1800

# EXPERIMENT CONTROL VARIABLES
# MUST BE CHANGED MANUALLY FOR EACH RUN (See INSTRUCTIONS at the end)
CHECKPOINT_MODE = "ADAPTIVE_CP" # Options: "NO_CP", "UNIFORM_CP", "ADAPTIVE_CP"
SAC_SCORE_THRESHOLD = 50        # Used only when CHECKPOINT_MODE is "ADAPTIVE_CP"

# --- MOCK PROFILING DATA (Simulates the output of a Memory Profiler) ---
# Higher score means more memory benefit from checkpointing.
MOCK_MEMORY_SCORES = {}
for i in range(N_LAYERS):
    # High scores for early/late layers, low scores in the middle
    if i < 4 or i >= N_LAYERS - 4:
        MOCK_MEMORY_SCORES[i] = random.randint(60, 95)
    elif N_LAYERS // 3 < i < N_LAYERS * 2 // 3:
        MOCK_MEMORY_SCORES[i] = random.randint(5, 20)
    else:
        MOCK_MEMORY_SCORES[i] = random.randint(40, 60)

# ---------------------------------------------------------------------
# MODEL DEFINITIONS (Simplified GPT-2 Like Architecture)
# ---------------------------------------------------------------------

class MockGPT2MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        inner_dim = embed_dim * 4
        self.c_fc = nn.Linear(embed_dim, inner_dim)
        self.c_proj = nn.Linear(inner_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class MockGPT2Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=16):
        super().__init__()
        num_heads = min(num_heads, embed_dim // 64)
        num_heads = max(num_heads, 8)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        attn_output, _ = self.attn(hidden_states, hidden_states, hidden_states)
        return self.resid_dropout(attn_output)

class MockGPT2Block(nn.Module):
    """A single Transformer block, target for FSDP and CP wrapping."""
    def __init__(self, embed_dim):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MockGPT2Attention(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MockGPT2MLP(embed_dim)

    def forward(self, hidden_states):
        attn_output = self.attn(self.ln_1(hidden_states))
        hidden_states = hidden_states + attn_output
        mlp_output = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + mlp_output
        return hidden_states

class MockGPT2Model(nn.Module):
    """The full GPT2 Model housing the sharded transformer blocks."""
    def __init__(self, n_layers, embed_dim, vocab_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(SEQ_LEN, embed_dim)
        self.h = nn.ModuleList([MockGPT2Block(embed_dim) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)

    def forward(self, input_ids):
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        return lm_logits


# ---------------------------------------------------------------------
# CHECKPOINTING LOGIC (Adaptive SAC Integration)
# ---------------------------------------------------------------------

def get_sac_decisions(n_layers: int, score_threshold: int, mode: str) -> list[bool]:
    """
    Determines which layers should be checkpointed based on the specified mode.
    """
    if mode == "NO_CP":
        return [False] * n_layers

    elif mode == "UNIFORM_CP":
        return [True] * n_layers

    elif mode == "ADAPTIVE_CP":
        decisions = []
        for i in range(n_layers):
            score = MOCK_MEMORY_SCORES.get(i, 0)
            # Checkpoint layer if its profiled memory score exceeds the threshold
            decisions.append(score >= score_threshold)
        return decisions

    else:
        raise ValueError(f"Unknown CHECKPOINT_MODE: {mode}")

def get_fsdp_wrap_policy():
    """
    Defines the FSDP auto-wrap policy for the transformer blocks.
    FIX: Uses transformer_auto_wrap_policy to avoid the AttributeError from FSDP.wrap.
    """
    return partial(
        transformer_auto_wrap_policy,
        # Specify the class that FSDP should automatically wrap
        transformer_layer_cls={MockGPT2Block},
    )

# ---------------------------------------------------------------------
# MAIN DISTRIBUTED TRAINING LOOP
# ---------------------------------------------------------------------

def init_dist(rank, world_size):
    """Initialize the distributed environment with the new port and timeout."""
    # MASTER_ADDR is localhost since we use mp.spawn on a single node
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = MASTER_PORT

    # FIX: Use datetime.timedelta instead of time.timedelta
    timeout = datetime.timedelta(seconds=COMMUNICATION_TIMEOUT_S)

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)

def main_worker(rank, world_size):
    # This worker process logic is what runs on each of the 4 GPUs
    init_dist(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Calculate local batch size
    local_batch_size = OPTIMAL_BATCH_SIZE // world_size

    # 1. Model Setup
    model = MockGPT2Model(N_LAYERS, EMBED_SIZE, VOCAB_SIZE).to(device)

    # 2. Adaptive SAC Decision
    sac_decisions = get_sac_decisions(N_LAYERS, SAC_SCORE_THRESHOLD, mode=CHECKPOINT_MODE)
    num_cp_layers = sum(sac_decisions)

    if rank == 0:
        print("\n" + "="*80)
        print(f"EXPERIMENT RUNNING: {CHECKPOINT_MODE} (Threshold={SAC_SCORE_THRESHOLD})")
        print(f"Distributed Setup: {world_size} GPUs, Local B: {local_batch_size}, Steps: {STEPS}")
        print(f"Model Size: {N_LAYERS} Layers, E={EMBED_SIZE}, Port: {MASTER_PORT}")
        print(f"Checkpointing Decision: {num_cp_layers}/{N_LAYERS} Layers Checkpointed.")
        print("="*80)

    # 3. Apply Checkpointing Wrapper (Manual Pre-Wrap: Fix for PyTorch < 2.1)
    # The SAC logic is applied here by conditionally wrapping the blocks.
    if num_cp_layers > 0 and CHECKPOINT_MODE != "NO_CP":
        new_h_blocks = nn.ModuleList()
        for i, block in enumerate(model.h):
            if sac_decisions[i]:
                # Wrap the block with the checkpoint wrapper
                wrapped_block = checkpoint_wrapper(
                    block,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT # Use the safe checkpoint implementation
                )
                new_h_blocks.append(wrapped_block)
            else:
                new_h_blocks.append(block)

        model.h = new_h_blocks

    # 4. FSDP Wrap (The outer wrapper)
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=get_fsdp_wrap_policy(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank
    )

    # Setup Optimizer and Loss
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- Training Simulation Loop ---

    # Mock data generation
    input_ids = torch.randint(0, VOCAB_SIZE, (local_batch_size, SEQ_LEN), device=device)
    labels = torch.randint(0, VOCAB_SIZE, (local_batch_size * SEQ_LEN,), device=device)

    step_times = []

    # Warmup steps
    fsdp_model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = fsdp_model(input_ids)
        loss = criterion(output.view(-1, VOCAB_SIZE), labels)
        loss.backward()
        optimizer.step()

    # BENCHMARKING LOOP
    for step in range(STEPS):
        torch.cuda.synchronize(device)
        start_time = time.time()

        optimizer.zero_grad(set_to_none=True)

        # Forward Pass
        output = fsdp_model(input_ids)
        loss = criterion(output.view(-1, VOCAB_SIZE), labels)

        # Backward Pass (FSDP AllGather/Reduce logic happens here)
        loss.backward()

        # Optimizer Step (FSDP AllReduce/Sharding update happens here)
        optimizer.step()

        torch.cuda.synchronize(device)
        step_times.append(time.time() - start_time)

    # Final Metrics Collection on Rank 0
    if rank == 0:
        avg_step_time = sum(step_times) / len(step_times)

        # Gather peak memory across all ranks for the true maximum
        max_mem_local = torch.cuda.max_memory_allocated(device) / (1024**3)
        max_mem_tensor = torch.tensor(max_mem_local, device=device)

        dist.all_reduce(max_mem_tensor, op=dist.ReduceOp.MAX)
        max_memory_allocated = max_mem_tensor.item()

        print("\n--- RESULTS SUMMARY (RANK 0) ---")
        print(f"CHECKPOINT_MODE: {CHECKPOINT_MODE}")
        print(f"Checkpoint Layers: {num_cp_layers}/{N_LAYERS}")
        print(f"Average Step Time: {avg_step_time:.4f} seconds/step")
        print(f"Peak Memory Allocated: {max_memory_allocated:.2f} GB (Max across all GPUs)")
        print("-----------------------")

    destroy_process_group()

if __name__ == '__main__':
    import torch.multiprocessing as mp

    # --- FIX: Environment Variables for NCCL/Communication Stability ---
    # 1. Disable InfiniBand (often fixes issues in local multi-GPU setups)
    os.environ['NCCL_IB_DISABLE'] = '1'
    # 2. Force using a standard socket network interface (like loopback)
    os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
    # ------------------------------------------------------------------

    # The old way to launch the distributed processes (compatible with torchrun)
    if torch.cuda.device_count() < NUM_GPUS:
        print(f"Error: Need {NUM_GPUS} GPUs, but only {torch.cuda.device_count()} found. Exiting.")
    else:
        mp.spawn(main_worker, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)