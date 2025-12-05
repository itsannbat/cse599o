"""
Corrected FSDP + Activation Checkpointing for GPT-2

Key fixes:
1. Use apply_activation_checkpointing with proper context preservation
2. Use CheckpointImpl.NO_REENTRANT to avoid metadata mismatch issues
3. Ensure deterministic recomputation
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class BlockActivationStats:
    layer_idx: int
    activation_bytes: int
    forward_time_ms: float


@dataclass
class FSDPUnit:
    unit_id: int
    start_layer: int
    end_layer: int
    total_activation_bytes: int


# ---------------------------------------------------------------------
# Adaptive Checkpoint Scheduler
# ---------------------------------------------------------------------

class AdaptiveCheckpointScheduler:
    """
    Compute- and memory-aware checkpoint scheduler.
    score(unit) = memory_saved_bytes - λ * recompute_cost_ms
    """

    def __init__(
        self,
        layer_stats: Dict[int, BlockActivationStats],
        shard_size: int = 4,
        savings_factor: float = 0.5,
        lambda_cost: float = 5.0,
    ):
        self.layer_stats = layer_stats
        self.num_layers = len(layer_stats)
        self.shard_size = shard_size
        self.savings_factor = savings_factor
        self.lambda_cost = lambda_cost
        self.units: List[FSDPUnit] = self._make_units()

    def _make_units(self) -> List[FSDPUnit]:
        units = []
        unit_id = 0
        for start in range(0, self.num_layers, self.shard_size):
            end = min(start + self.shard_size - 1, self.num_layers - 1)

            total_bytes = 0
            total_time_ms = 0.0

            for layer_idx in range(start, end + 1):
                total_bytes += self.layer_stats[layer_idx].activation_bytes
                total_time_ms += self.layer_stats[layer_idx].forward_time_ms

            unit = FSDPUnit(
                unit_id=unit_id,
                start_layer=start,
                end_layer=end,
                total_activation_bytes=total_bytes,
            )
            unit.total_time_ms = total_time_ms
            units.append(unit)
            unit_id += 1

        return units

    def _baseline_peak_memory(self) -> int:
        return sum(s.activation_bytes for s in self.layer_stats.values())

    def schedule(self, budget_ratio: float):
        baseline = self._baseline_peak_memory()
        target = int(baseline * budget_ratio)

        scored_units = []
        for u in self.units:
            memory_saved = int(self.savings_factor * u.total_activation_bytes)
            compute_penalty = self.lambda_cost * u.total_time_ms
            score = memory_saved - compute_penalty
            scored_units.append((u, score, memory_saved, compute_penalty))

        scored_units.sort(key=lambda x: x[1], reverse=True)

        chosen = []
        current_mem = baseline

        for u, score, mem_saved, comp_cost in scored_units:
            if current_mem <= target:
                break
            if score < 0:
                continue
            chosen.append(u)
            current_mem -= mem_saved

        # Create set of layer indices to checkpoint
        checkpoint_layers = set()
        for u in chosen:
            for layer_idx in range(u.start_layer, u.end_layer + 1):
                checkpoint_layers.add(layer_idx)

        debug = {
            "baseline_bytes": baseline,
            "target_bytes": target,
            "final_estimated_bytes": current_mem,
            "chosen_units": chosen,
            "unit_scores": scored_units,
        }
        return checkpoint_layers, debug


# ---------------------------------------------------------------------
# FSDP Activation Profiler
# ---------------------------------------------------------------------

class FSDPActivationProfiler:
    """
    Activation profiler that works with FSDP-wrapped modules.
    """

    def __init__(self, fsdp_model):
        self.model = fsdp_model
        self.layer_stats: Dict[int, BlockActivationStats] = {}
        self._handles = []

    def _register_hooks(self):
        """
        Attach forward hooks to each FSDP-wrapped transformer block.
        """
        for idx, block in enumerate(self.model.transformer.h):
            time_key = f"_tstart_{idx}"

            def pre_hook(mod, inputs, idx=idx, key=time_key):
                mod.__dict__[key] = time.time()

            def post_hook(mod, inputs, output, idx=idx, key=time_key):
                t0 = mod.__dict__.pop(key, None)
                t1 = time.time()
                elapsed = (t1 - t0) * 1000.0 if t0 is not None else 0.0

                # Handle tuple outputs
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output

                act_bytes = out.numel() * out.element_size()

                self.layer_stats[idx] = BlockActivationStats(
                    layer_idx=idx,
                    activation_bytes=act_bytes,
                    forward_time_ms=elapsed,
                )

            h1 = block.register_forward_pre_hook(pre_hook)
            h2 = block.register_forward_hook(post_hook)
            self._handles.extend([h1, h2])

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def profile(self, input_ids):
        """
        Run a single forward pass and collect activation stats.
        """
        self.model.eval()
        self._register_hooks()

        with torch.no_grad():
            _ = self.model(input_ids)

        self._remove_hooks()
        return self.layer_stats


# ---------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------

def setup_distributed():
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            rank=rank,
            world_size=world_size,
        )

    return dist.get_rank(), dist.get_world_size()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------
# FSDP wrapping with activation checkpointing
# ---------------------------------------------------------------------

def wrap_gpt2_blocks_with_fsdp(
    model: GPT2LMHeadModel,
    device: torch.device,
    checkpoint_layers: Optional[Set[int]] = None,
):
    """
    Wrap each GPT-2 transformer block as its own FSDP unit.
    Optionally apply activation checkpointing to specific layers.

    CRITICAL: Uses CheckpointImpl.NO_REENTRANT to avoid metadata mismatch
    issues with attention mechanisms.
    """
    assert dist.is_initialized(), "Distributed must be initialized"

    for i, block in enumerate(model.transformer.h):
        block.to(device)

        if device.type == "cuda":
            fsdp_block = FSDP(
                block,
                device_id=device,
                use_orig_params=True,
            )
        else:
            fsdp_block = FSDP(
                block,
                use_orig_params=True,
            )

        model.transformer.h[i] = fsdp_block

    # Apply activation checkpointing if specified
    if checkpoint_layers is not None and len(checkpoint_layers) > 0:

        def check_fn(submodule):
            # Check if this is one of the blocks we want to checkpoint
            for idx, block in enumerate(model.transformer.h):
                if submodule is block and idx in checkpoint_layers:
                    return True
            return False

        # CRITICAL FIX: Use NO_REENTRANT implementation
        # This avoids the metadata mismatch error with attention tensors
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn,
        )

    return model


# ---------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------

def run_step(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
) -> float:
    """
    Simple training step - let FSDP handle everything.
    """
    # IMPORTANT: Don't use past_key_values - they cause non-determinism
    outputs = model(input_ids, labels=input_ids, use_cache=False)
    loss = outputs.loss
    loss.backward()
    return float(loss.item())


def benchmark_mode(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    steps: int = 4,
) -> Tuple[float, int]:
    """
    Benchmark the model with current checkpointing configuration.
    """
    model.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)

    times = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        _ = run_step(model, input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        times.append(t1 - t0)

        # Take optimizer step to match real training
        optimizer.step()

    avg_time = sum(times) / len(times)
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    return avg_time, peak_mem


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    rank, world_size = setup_distributed()
    is_main = (rank == 0)

    if torch.cuda.is_available():
        device = torch.device("cuda", rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    if is_main:
        print(f"[Rank {rank}] World size: {world_size}, device: {device}")

    # Create model
    cfg = GPT2Config(
        n_embd=768,
        n_layer=24,
        n_head=12,
        n_positions=512,
        vocab_size=50000,
    )

    B, T = 2, 128

    # ---- Test 1: No checkpointing ----
    if is_main:
        print("\n=== Test 1: No Activation Checkpointing ===")

    model_no_cp = GPT2LMHeadModel(cfg).to(device)
    model_no_cp = wrap_gpt2_blocks_with_fsdp(model_no_cp, device, checkpoint_layers=None)
    optimizer_no_cp = torch.optim.AdamW(model_no_cp.parameters(), lr=1e-3)

    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    avg_time_no_cp, peak_mem_no_cp = benchmark_mode(model_no_cp, input_ids, optimizer_no_cp)

    if is_main:
        print(f"No-CP: {avg_time_no_cp:.6f} sec, {peak_mem_no_cp/1024**2:.2f} MB")

    del model_no_cp, optimizer_no_cp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Test 2: Profile activations ----
    if is_main:
        print("\n=== Test 2: Profiling Activations ===")

    model_profile = GPT2LMHeadModel(cfg).to(device)
    model_profile = wrap_gpt2_blocks_with_fsdp(model_profile, device, checkpoint_layers=None)

    profiler = FSDPActivationProfiler(model_profile)
    layer_stats = profiler.profile(input_ids)

    if is_main:
        print(f"\nPer-layer activation stats:")
        for idx in sorted(layer_stats.keys()):
            s = layer_stats[idx]
            print(f"  Layer {idx:02d}: {s.activation_bytes/1024**2:.3f} MB, "
                  f"{s.forward_time_ms:.3f} ms")

    del model_profile
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Test 3: Adaptive checkpointing ----
    if is_main:
        print("\n=== Test 3: Adaptive Checkpointing ===")

    scheduler = AdaptiveCheckpointScheduler(
        layer_stats=layer_stats,
        shard_size=4,
        savings_factor=0.5,
        lambda_cost=5.0,
    )

    checkpoint_layers, dbg = scheduler.schedule(budget_ratio=0.7)

    if is_main:
        print(f"Baseline activation memory: {dbg['baseline_bytes']/1024**2:.3f} MB")
        print(f"Target activation memory:   {dbg['target_bytes']/1024**2:.3f} MB")
        print(f"Estimated final memory:     {dbg['final_estimated_bytes']/1024**2:.3f} MB")
        print(f"Checkpointing {len(checkpoint_layers)} layers: {sorted(checkpoint_layers)}")

    model_adaptive = GPT2LMHeadModel(cfg).to(device)
    model_adaptive = wrap_gpt2_blocks_with_fsdp(model_adaptive, device, checkpoint_layers)
    optimizer_adaptive = torch.optim.AdamW(model_adaptive.parameters(), lr=1e-3)

    avg_time_adaptive, peak_mem_adaptive = benchmark_mode(model_adaptive, input_ids, optimizer_adaptive)

    if is_main:
        print(f"Adaptive-CP: {avg_time_adaptive:.6f} sec, {peak_mem_adaptive/1024**2:.2f} MB")
        print(f"\n=== Summary ===")
        print(f"Memory saved: {(peak_mem_no_cp - peak_mem_adaptive)/1024**2:.2f} MB "
              f"({100*(peak_mem_no_cp - peak_mem_adaptive)/peak_mem_no_cp:.1f}% reduction)")
        print(f"Time overhead: {(avg_time_adaptive - avg_time_no_cp)*1000:.2f} ms "
              f"({100*(avg_time_adaptive - avg_time_no_cp)/avg_time_no_cp:.1f}% slower)")

    # ---- Test 4: Uniform checkpointing for comparison ----
    if is_main:
        print("\n=== Test 4: Uniform Checkpointing (all layers) ===")

    del model_adaptive, optimizer_adaptive
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_layers = set(range(cfg.n_layer))
    model_uniform = GPT2LMHeadModel(cfg).to(device)
    model_uniform = wrap_gpt2_blocks_with_fsdp(model_uniform, device, all_layers)
    optimizer_uniform = torch.optim.AdamW(model_uniform.parameters(), lr=1e-3)

    avg_time_uniform, peak_mem_uniform = benchmark_mode(model_uniform, input_ids, optimizer_uniform)

    if is_main:
        print(f"Uniform-CP: {avg_time_uniform:.6f} sec, {peak_mem_uniform/1024**2:.2f} MB")
        print(f"Adaptive vs Uniform time: {(avg_time_adaptive - avg_time_uniform)*1000:.2f} ms faster")

    cleanup_distributed()


if __name__ == "__main__":
    main()