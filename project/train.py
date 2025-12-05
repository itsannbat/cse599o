# ---------------------------------------------------------------------
# FSDP-aware Activation Profiler
# ---------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, Optional, Set, List, Tuple
from functools import partial
import os
import time
import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from enum import Enum, auto

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


@dataclass
class FSDPUnit:
    unit_id: int
    start_layer: int
    end_layer: int
    total_activation_bytes: int

class CPPolicy(Enum):
    NO_CP = auto()
    UNIFORM = auto()
    TOY_BOUNDARY = auto()
    ADAPTIVE = auto()

@dataclass
class BlockActivationStats:
    layer_idx: int
    activation_bytes: int
    forward_time_ms: float

def make_index_based_check_fn(block_type, selected_indices: set[int]):
    """
    Returns a check_fn(submodule) -> bool that checkpoints exactly the
    layers whose 0-based index is in selected_indices.
    """
    block_idx = 0

    def check_fn(submodule):
        nonlocal block_idx
        if isinstance(submodule, block_type):
            should_ckpt = block_idx in selected_indices
            block_idx += 1
            return should_ckpt
        return False

    return check_fn


def make_uniform_check_fn(block_type):
    def check_fn(submodule):
        return isinstance(submodule, block_type)
    return check_fn


def make_no_cp_check_fn(block_type):
    def check_fn(_submodule):
        return False
    return check_fn


def apply_cp_policy(
    model,
    block_type,
    policy: CPPolicy,
    *,
    adaptive_selected_indices: set[int] | None = None,
    toy_boundary_idx: int | None = None,
):
    """
    Apply activation checkpointing policy to an (FSDP-wrapped) model.

    block_type: the layer class to treat as "blocks" (e.g., GPT2Block).
    """

    if policy == CPPolicy.NO_CP:
        check_fn = make_no_cp_check_fn(block_type)

    elif policy == CPPolicy.UNIFORM:
        check_fn = make_uniform_check_fn(block_type)

    elif policy == CPPolicy.TOY_BOUNDARY:
        assert toy_boundary_idx is not None, "toy_boundary_idx required for TOY_BOUNDARY"
        # checkpoint [0..toy_boundary_idx-1]
        selected = set(range(toy_boundary_idx))
        check_fn = make_index_based_check_fn(block_type, selected)

    elif policy == CPPolicy.ADAPTIVE:
        assert adaptive_selected_indices is not None, "adaptive_selected_indices required for ADAPTIVE"
        check_fn = make_index_based_check_fn(block_type, adaptive_selected_indices)

    else:
        raise ValueError(f"Unsupported CP policy: {policy}")

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )


class AdaptiveCheckpointScheduler:
    """
    Compute- and memory-aware checkpoint scheduler.

    For each FSDPUnit:
        - memory_saved = savings_factor * total_activation_bytes
        - recompute_cost_ms = gamma * forward_time_ms
        - score = memory_saved - λ * recompute_cost_ms

    Higher score → better candidate for checkpointing.
    """

    def __init__(
        self,
        layer_stats: Dict[int, BlockActivationStats],
        shard_size: int = 4,
        savings_factor: float = 1.0,      # default: assume full savings
        lambda_cost: float = 5.0,
        gamma_recompute_factor: float = 2.0,  # ≈ FWD + BWD recompute
    ):
        self.layer_stats = layer_stats
        self.num_layers = len(layer_stats)
        self.shard_size = shard_size
        self.savings_factor = savings_factor
        self.lambda_cost = lambda_cost
        self.gamma_recompute_factor = gamma_recompute_factor
        self.units: List[FSDPUnit] = self._make_units()

    # -------------------- build FSDP units --------------------
    def _make_units(self) -> List[FSDPUnit]:
        units = []
        unit_id = 0

        for start in range(0, self.num_layers, self.shard_size):
            end = min(start + self.shard_size - 1, self.num_layers - 1)

            total_bytes = 0
            total_forward_time_ms = 0.0

            for layer_idx in range(start, end + 1):
                stat = self.layer_stats[layer_idx]
                total_bytes += stat.activation_bytes
                total_forward_time_ms += stat.forward_time_ms

            unit = FSDPUnit(
                unit_id=unit_id,
                start_layer=start,
                end_layer=end,
                total_activation_bytes=total_bytes,
            )

            unit.total_time_ms = total_forward_time_ms
            unit.recompute_cost_ms = (
                self.gamma_recompute_factor * total_forward_time_ms
            )

            units.append(unit)
            unit_id += 1

        return units

    # ---------------- baseline peak memory --------------------
    def _baseline_peak_memory(self) -> int:
        # Better approximation: peak ≈ max activation among layers
        return sum(stat.activation_bytes for stat in self.layer_stats.values())

    # ---------------------- scheduling ------------------------
    def schedule(self, budget_ratio: float):
        baseline = self._baseline_peak_memory()
        target = int(baseline * budget_ratio)

        scored_units = []
        for u in self.units:
            memory_saved = self.savings_factor * u.total_activation_bytes
            compute_penalty = self.lambda_cost * u.recompute_cost_ms
            score = memory_saved - compute_penalty
            scored_units.append((u, score, memory_saved, compute_penalty))

        # Higher score is better
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

        # Fallback: if we still exceed target and picked nothing,
        # choose the unit with max memory_saved.
        if current_mem > target and not chosen:
            best_unit, _, best_mem_saved, _ = max(
                scored_units, key=lambda x: x[2]
            )
            chosen = [best_unit]
            current_mem = baseline - best_mem_saved

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
        Uses CUDA events + NVTX for accurate timing.
        """
        for idx, block in enumerate(self.model.transformer.h):
            if not hasattr(block, "_fsdp_wrapped_module"):
                raise RuntimeError(
                    f"Expected FSDP-wrapped block at index {idx}, "
                    "but FSDP wrapper structure not found."
                )

            inner = block._fsdp_wrapped_module  # the actual GPT2Block

            start_key = f"_start_event_{idx}"
            end_key = f"_end_event_{idx}"

            def pre_hook(mod, inputs, idx=idx):
                mod._t0 = time.time()

            def post_hook(mod, inputs, output, idx=idx):
                t1 = time.time()
                elapsed = (t1 - mod._t0) * 1000.0
                del mod._t0

                # extract tensors
                out = output[0] if isinstance(output, tuple) else output
                act_bytes = out.numel() * out.element_size()

                self.layer_stats[idx] = BlockActivationStats(
                    layer_idx=idx,
                    activation_bytes=act_bytes,
                    forward_time_ms=elapsed,
                )

            # register hooks
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

def reset_cuda():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def benchmark_mode(model, input_ids, mode, boundaries=None, steps=100, warmup_steps=40): # <-- New: warmup_steps
    reset_cuda()

    # --- WARM-UP LOOP ---
    for _ in range(warmup_steps):
        # We only care about the time/mem from the actual test,
        # but we need to run the full loss+backward to warm up all kernels.
        _ = run_step(model, input_ids, mode=mode, boundaries=boundaries)

    # Clear memory/stats before the timed run - CRITICAL for accurate measurement
    reset_cuda()
    # Reset peak memory stats right before timed loop to ensure clean measurement
    torch.cuda.reset_peak_memory_stats()

    # --- TIMED LOOP ---
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.time()

        loss = run_step(model, input_ids, mode=mode, boundaries=boundaries)

        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

        # NOTE: You MUST do the optimizer step inside the timed loop
        # or the memory peak from the optimizer step will not be included.
        # However, your current benchmark_mode does NOT include the optimizer step.
        # If you intend to measure ONLY FWD+BWD, this is fine.
        # If you intend to measure FWD+BWD+Optimizer, you must add it here.
        # For a fair comparison with real training, you should include it.
        # If you don't, the memory will only show the FWD/BWD peak.

    # Synchronize and measure peak memory BEFORE any cleanup
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated()
    avg_time = sum(times) / len(times)

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

def build_fsdp_gpt2(cfg: GPT2Config, device: torch.device) -> GPT2LMHeadModel:
    """
    Construct a fresh GPT2LMHeadModel, move it to device,
    and wrap its blocks with FSDP.
    """
    base_model = GPT2LMHeadModel(cfg).to(device)
    model = wrap_gpt2_blocks_with_fsdp(base_model, device=device)
    return model


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # ---------------- Distributed Setup ----------------
    rank, world_size = setup_distributed()
    is_main = (rank == 0)

    if torch.cuda.is_available():
        device = torch.device("cuda", rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    if is_main:
        print(f"[Rank {rank}] World size: {world_size}, device: {device}")

    # ---------------- Model Config ----------------
    cfg = GPT2Config(
        n_embd=768,
        n_layer=24,
        n_head=12,
        n_positions=512,
        vocab_size=50000,
    )

    if is_main:
        print(f"[Rank {rank}] GPT2 Config: layers={cfg.n_layer}, embd={cfg.n_embd}")

    # ---------------- Build Profiling Model ----------------
    profile_model = build_fsdp_gpt2(cfg, device=device)

    if is_main:
        print(f"[Rank {rank}] Transformer block type: "
              f"{type(profile_model.transformer.h[0]._fsdp_wrapped_module)}")

    # ---------------- Input Tokens ----------------
    B, T = 2, 128
    vocab_size = cfg.vocab_size
    input_ids = torch.randint(0, vocab_size, (B, T), device=device)

    # ---------------- 1. Activation Profiling ----------------
    if is_main:
        print("\n[Profiling Activations Under FSDP]")

    profiler = FSDPActivationProfiler(profile_model)
    layer_stats = profiler.profile(input_ids)

    if is_main:
        for idx in sorted(layer_stats.keys()):
            s = layer_stats[idx]
            print(
                f"Layer {idx:02d}: "
                f"act={s.activation_bytes/1024**2:.3f} MB, "
                f"time={s.forward_time_ms:.3f} ms"
            )

    # ---------------- 2. Adaptive Scheduler ----------------
    shard_size = 4
    budget_ratio = 0.7
    savings_factor = 0.5

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

        print("Chosen units for Adaptive-CP:")
        for u in dbg["chosen_units"]:
            print(f"  Unit {u.unit_id}: layers[{u.start_layer}-{u.end_layer}]")

        print("Adaptive checkpoint boundaries:", sorted(adaptive_boundaries))

    # ---------------- Toy Boundary (simple baseline) ----------------
    num_layers = cfg.n_layer
    shard_size_toy = 4
    toy_boundaries = set(
        i for i in range(shard_size_toy - 1, num_layers, shard_size_toy)
    )

    if is_main:
        print(f"\n[Rank {rank}] Toy checkpoint boundaries: {sorted(toy_boundaries)}")

    # ---------------- Cleanup profiling model ----------------
    del profiler
    del profile_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ---------------- 3. Benchmark All Modes ----------------
    modes = [
        ("no_cp",        None,                 "No-CP"),
        ("uniform_cp",   None,                 "Uniform-CP"),
        ("adaptive_cp",  adaptive_boundaries,  "Adaptive-CP"),
        ("boundary_cp",  toy_boundaries,       "ToyBoundary-CP"),
    ]

    results = {}

    for mode, boundaries, label in modes:

        if is_main:
            print(f"\n[Rank {rank}] ===== Benchmarking {label} =====")

        # Clear memory state before building the model for this mode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # ---- Fresh model + optimizer for each mode (order-independent) ----
        model_for_mode = build_fsdp_gpt2(cfg, device=device)
        optimizer_for_mode = torch.optim.AdamW(model_for_mode.parameters(), lr=1e-3)

        # Reset peak memory stats AFTER model is built and BEFORE benchmarking
        # This ensures model allocation is not included in the measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # ---- Full-step benchmark ----
        avg_time, peak_mem = benchmark_mode(
            model=model_for_mode,
            input_ids=input_ids,
            mode=mode,
            boundaries=boundaries,
            steps=100,
            warmup_steps=40
        )

        results[label] = (avg_time, peak_mem)

        if is_main:
            print(
                f"[Rank {rank}] {label:15s} "
                f"avg_step_time={avg_time:.6f} sec "
                f"peak_mem={peak_mem/1024**2:.2f} MB"
            )

        # ---- Optional correctness check: one optimizer step ----
        # Note: This runs AFTER memory measurement, so it doesn't affect peak_mem
        optimizer_for_mode.zero_grad(set_to_none=True)
        _ = run_step(model_for_mode, input_ids, mode=mode, boundaries=boundaries)
        optimizer_for_mode.step()

        # Cleanup this mode's model - ensure FSDP state is cleared
        del optimizer_for_mode
        # FSDP cleanup: ensure all FSDP modules are properly destroyed
        # Note: FSDP modules should be properly cleaned up when model is deleted
        del model_for_mode

        # Aggressive cleanup to ensure all memory is freed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak memory stats after cleanup to ensure next mode starts fresh
            torch.cuda.reset_peak_memory_stats()

            # Small delay to allow CUDA to fully release memory
            # This helps prevent memory fragmentation issues
            time.sleep(0.1)

    # ---------------- Final Summary ----------------
    if is_main:
        print("\n[Rank 0] Summary (Time + Memory):")
        for label, (t, mem) in results.items():
            print(f"  {label:15s}: {t:.6f} sec, {mem/1024**2:.2f} MB")

    cleanup_distributed()


if __name__ == "__main__":
    main()
