# ---------------------------------------------------------------------
# FSDP-aware Activation Profiler
# ---------------------------------------------------------------------
import time
from dataclasses import dataclass
from typing import Dict

@dataclass
class BlockActivationStats:
    layer_idx: int
    activation_bytes: int
    forward_time_ms: float

# ---------------------------------------------------------------------
# Adaptive, memory-budget-based checkpoint scheduler
# ---------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

@dataclass
class FSDPUnit:
    unit_id: int
    start_layer: int
    end_layer: int
    total_activation_bytes: int


class AdaptiveCheckpointScheduler:
    """
    Compute- and memory-aware checkpoint scheduler.

    score(unit) = memory_saved_bytes - λ * recompute_cost_ms

    Lower recompute cost → preferred
    Larger memory savings → preferred
    """

    def __init__(
        self,
        layer_stats: Dict[int, BlockActivationStats],
        shard_size: int = 4,
        savings_factor: float = 0.5,
        lambda_cost: float = 5.0,   # weight for compute penalty (tunable)
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

            units.append(
                FSDPUnit(
                    unit_id=unit_id,
                    start_layer=start,
                    end_layer=end,
                    total_activation_bytes=total_bytes,
                )
            )
            units[-1].total_time_ms = total_time_ms  # attach compute cost
            unit_id += 1

        return units

    def _baseline_peak_memory(self) -> int:
        return sum(s.activation_bytes for s in self.layer_stats.values())

    def schedule(self, budget_ratio: float):
        baseline = self._baseline_peak_memory()
        target = int(baseline * budget_ratio)

        # compute-aware score
        scored_units = []
        for u in self.units:
            memory_saved = int(self.savings_factor * u.total_activation_bytes)
            compute_penalty = self.lambda_cost * u.total_time_ms
            score = memory_saved - compute_penalty
            scored_units.append((u, score, memory_saved, compute_penalty))

        # higher score = better
        scored_units.sort(key=lambda x: x[1], reverse=True)

        chosen = []
        current_mem = baseline

        for u, score, mem_saved, comp_cost in scored_units:
            if current_mem <= target:
                break
            if score < 0:
                continue  # too expensive to recompute
            chosen.append(u)
            current_mem -= mem_saved

        boundaries = {u.end_layer for u in chosen}

        debug = {
            "baseline_bytes": baseline,
            "target_bytes": target,
            "final_estimated_bytes": current_mem,
            "chosen_units": chosen,
            "unit_scores": scored_units,
        }
        return boundaries, debug


class FSDPActivationProfiler:
    """
    Activation profiler for FSDP(GPT2Block).

    ✔ Works on FSDP-wrapped modules (FullyShardedDataParallel)
    ✔ Handles tuple outputs (GPT-2 returns (hidden_states, ...))
    ✔ Generates per-layer:
        - activation size in bytes
        - forward compute time
    ✔ Uses forward hooks on the *inner module* (module.module)
      because FSDP wraps the block inside an internal container.
    """

    def __init__(self, fsdp_model):
        """
        fsdp_model is the GPT2LMHeadModel with its blocks wrapped in FSDP.
        """
        self.model = fsdp_model
        self.layer_stats: Dict[int, BlockActivationStats] = {}
        self._handles = []  # store hook handles to remove later

    def _register_hooks(self):
        """
        Attach forward hooks to each FSDP-wrapped transformer block.

        IMPORTANT:
        - FSDP wraps the underlying block in module._fsdp_wrapped_module
          so we must hook that inner module.
        """
        for idx, block in enumerate(self.model.transformer.h):
            if not hasattr(block, "_fsdp_wrapped_module"):
                raise RuntimeError(
                    f"Expected FSDP-wrapped block at index {idx}, "
                    "but FSDP wrapper structure not found."
                )

            inner = block._fsdp_wrapped_module  # the actual GPT2Block

            time_key = f"_tstart_{idx}"

            def pre_hook(mod, inputs, idx=idx, key=time_key):
                mod.__dict__[key] = time.time()

            def post_hook(mod, inputs, output, idx=idx, key=time_key):
                t0 = mod.__dict__.pop(key, None)
                t1 = time.time()
                elapsed = (t1 - t0) * 1000.0 if t0 is not None else 0.0

                # Get main tensor (hidden_states)
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

            # Register hooks on the *inner* module
            h1 = inner.register_forward_pre_hook(pre_hook)
            h2 = inner.register_forward_hook(post_hook)
            self._handles.extend([h1, h2])

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def profile(self, input_ids):
        """
        Run a single forward pass and collect activation stats.

        Returns: dict[layer_idx → BlockActivationStats]
        """
        self.model.eval()
        self._register_hooks()

        with torch.no_grad():
            _ = self.model(input_ids)

        self._remove_hooks()
        return self.layer_stats


"""
fsdp_gpt2_fsdp_blocks.py

Step 1: Real FSDP wrapping around GPT-2 transformer blocks.

What this script does:
- Initializes torch.distributed (single- or multi-GPU).
- Builds a small GPT-2 model from HuggingFace transformers.
- Wraps EACH transformer block (model.transformer.h[i]) with FSDP,
  so each block is its own FSDP unit.
- Runs a few training steps in three modes:
    * "no_cp"       : no activation checkpointing
    * "uniform_cp"  : checkpoint every block
    * "boundary_cp" : checkpoint only at some layer indices (toy example)
- Prints step timings from rank 0.

This is the foundation we can later extend with:
- activation profiling,
- adaptive, budget-based schedules,
- FSDP-aware boundary selection from real units.
"""

import os
import time
from typing import Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


# ---------------------------------------------------------------------
# Distributed setup / teardown
# ---------------------------------------------------------------------

def setup_distributed():
    if not dist.is_initialized():
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # force a safe port
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            rank=rank,
            world_size=world_size,
        )

    return dist.get_rank(), dist.get_world_size()




def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------

def _call_block(block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Call a GPT-2 block and return the main hidden-state tensor.
    (GPT2Block can return a tuple; we just want the tensor.)
    """
    output = block(hidden_states)
    return output[0] if isinstance(output, tuple) else output


def run_step(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    mode: str = "no_cp",
    boundaries: Optional[Set[int]] = None,
) -> float:
    """
    mode:
      - "no_cp"
      - "uniform_cp"
      - "boundary_cp"
      - "adaptive_cp"   <-- NEW (same mechanics as boundary, different schedule)
    """
    transformer = model.transformer
    lm_head = model.lm_head

    hidden_states = transformer.wte(input_ids)
    hidden_states = transformer.drop(hidden_states)

    for i, block in enumerate(transformer.h):
        if mode == "uniform_cp":
            hidden_states = checkpoint(_call_block, block, hidden_states)

        elif mode in ("boundary_cp", "adaptive_cp"):
            if boundaries is not None and i in boundaries:
                hidden_states = checkpoint(_call_block, block, hidden_states)
            else:
                hidden_states = _call_block(block, hidden_states)

        else:  # "no_cp"
            hidden_states = _call_block(block, hidden_states)

    hidden_states = transformer.ln_f(hidden_states)
    logits = lm_head(hidden_states)

    loss = logits.mean()
    loss.backward()
    return float(loss.item())



def benchmark_mode(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    mode: str,
    boundaries: Optional[Set[int]] = None,
    steps: int = 4,
) -> Tuple[float, int]:
    """
    Returns:
       avg_step_time (float)
       peak_memory_bytes (int)
    """
    assert mode in {"no_cp", "uniform_cp", "boundary_cp", "adaptive_cp"}
    model.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_ids)

    times = []

    for _ in range(steps):
        model.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        _ = run_step(model, input_ids, mode=mode, boundaries=boundaries)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        times.append(t1 - t0)

    avg_time = sum(times) / len(times)

    # Peak memory
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        peak_mem = 0

    return avg_time, peak_mem


# ---------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------

def wrap_gpt2_blocks_with_fsdp(model: GPT2LMHeadModel, device: torch.device):
    """
    Wrap EACH GPT-2 transformer block as its own FSDP unit.

    This is a simple nested-FSDP scheme:
      - The top-level GPT2LMHeadModel stays as a normal nn.Module.
      - Each model.transformer.h[i] becomes FSDP(block_i, ...).

    This matches the "FSDP unit per block" concept from our project,
    and later we can read these units as the real boundaries.
    """
    # Get the default process group (created by setup_distributed)
    assert dist.is_initialized(), "Distributed must be initialized before FSDP wrapping."
    process_group = dist.group.WORLD

    for i, block in enumerate(model.transformer.h):
        # Put block on the correct device first
        block.to(device)

        # Wrap with FSDP. use_orig_params=True is recommended for nested FSDP.
        if device.type == "cuda":
            fsdp_block = FSDP(
                block,
                process_group=process_group,
                device_id=device,
                use_orig_params=True,
            )
        else:
            fsdp_block = FSDP(
                block,
                process_group=process_group,
                use_orig_params=True,
            )

        model.transformer.h[i] = fsdp_block

    return model


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

    # ---- Model ----
    cfg = GPT2Config(
        n_embd=768,        # from 256 → 768
        n_layer=24,        # from 12 → 24
        n_head=12,
        n_positions=512,  # from 128 → 512
        vocab_size=50000,
    )
    base_model = GPT2LMHeadModel(cfg).to(device)

    if is_main:
        print(f"[Rank {rank}] Transformer block type: "
              f"{type(base_model.transformer.h[0])}")

    model = wrap_gpt2_blocks_with_fsdp(base_model, device=device)

    # ---- Input sample ----
    B, T = 2, 128
    vocab_size = cfg.vocab_size
    input_ids = torch.randint(0, vocab_size, (B, T), device=device)

    # ---- 1. Activation Profiling ----
    if is_main:
        print("\n[Profiling Activations Under FSDP]")

    profiler = FSDPActivationProfiler(model)
    layer_stats = profiler.profile(input_ids)

    if is_main:
        for idx in sorted(layer_stats.keys()):
            s = layer_stats[idx]
            print(
                f"Layer {idx:02d}: "
                f"act={s.activation_bytes/1024**2:.3f} MB, "
                f"time={s.forward_time_ms:.3f} ms"
            )

    # ---- 2. Adaptive scheduler based on memory budget ----
    shard_size = 4          # treat groups of 4 layers as one "unit"
    budget_ratio = 0.7      # target 70% of baseline activation memory
    savings_factor = 0.5    # assume ~50% savings when checkpointing a unit

    scheduler = AdaptiveCheckpointScheduler(
        layer_stats=layer_stats,
        shard_size=shard_size,
        savings_factor=savings_factor,
    )
    adaptive_boundaries, dbg = scheduler.schedule(budget_ratio=budget_ratio)

    if is_main:
      print("\n[Adaptive Scheduler Debug]")
      print(f"Baseline activation: {dbg['baseline_bytes']/1024**2:.3f} MB")
      print(f"Target activation:   {dbg['target_bytes']/1024**2:.3f} MB")
      print(f"Estimated final:     {dbg['final_estimated_bytes']/1024**2:.3f} MB")

      print("\nUnit scores (memory MB, compute ms, score):")
      for u, score, mem_saved, comp_cost in dbg["unit_scores"]:
          print(
              f"  U{u.unit_id}: layers[{u.start_layer}-{u.end_layer}] "
              f"mem_saved={mem_saved/1024**2:.3f} MB  "
              f"comp_cost={comp_cost:.3f} ms  "
              f"score={score:.3f}"
          )

      print("Chosen units:")
      for u in dbg["chosen_units"]:
          print(f"  Unit {u.unit_id}: layers[{u.start_layer}-{u.end_layer}]")

      print("Adaptive checkpoint boundaries:", sorted(adaptive_boundaries))


    # (Optional) keep your old toy boundaries for comparison
    num_layers = cfg.n_layer
    shard_size_toy = 4
    toy_boundaries = set(
        i for i in range(shard_size_toy - 1, num_layers, shard_size_toy)
    )
    if is_main:
        print(f"\n[Rank {rank}] Checkpoint boundaries (toy): "
              f"{sorted(toy_boundaries)}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # ---- 3. Benchmark modes ----
    results = {}

    # (mode, boundaries, label)
    modes = [
        ("no_cp",        None,            "No-CP"),
        ("uniform_cp",   None,            "Uniform-CP"),
        ("boundary_cp",  toy_boundaries,  "ToyBoundary-CP"),
        ("adaptive_cp",  adaptive_boundaries, "Adaptive-CP"),
    ]

    results = {}

    for mode, boundaries, label in modes:
        avg_time, peak_mem = benchmark_mode(
            model=model,
            input_ids=input_ids,
            mode=mode,
            boundaries=boundaries,
            steps=4,
        )

        results[label] = (avg_time, peak_mem)

        if is_main:
            print(f"[Rank {rank}] Mode={label:15s} "
                  f"avg_step_time={avg_time:.6f} sec "
                  f"peak_mem={peak_mem/1024**2:.2f} MB")

        # optimizer step
        optimizer.zero_grad(set_to_none=True)
        loss = run_step(model, input_ids, mode=mode, boundaries=boundaries)
        optimizer.step()


    if is_main:
        print("\n[Rank 0] Summary (Time + Memory):")
        for label, (t, mem) in results.items():
            print(f"  {label:15s}: {t:.6f} sec, {mem/1024**2:.2f} MB")


    cleanup_distributed()

if __name__ == "__main__":
    main()
