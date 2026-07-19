"""Extended-roofline GEMM backend (the default kernel-time model).

Replaces RAPID-LLM's per-GEMM kernel time with an extended-roofline model:
shape-dependent effective utilization from CTA tiling, wave quantization,
L2 reuse, per-level bandwidth limits, and an explicit SMEM/MMA/L2/DRAM
wave-pipeline timeline. Everything else — tile selection, memory access
accounting, launch overhead, communication, scheduling — stays native.

The model originates from eyao600/op-model @ 3025422 (which includes the
SMEM-prologue pipelining fix: under a multi-stage cp.async pipeline the
steady-state k-group runs at max(smem, math), not smem + math). It is
implemented here as first-class repo code; there is no external package.

Semantics:
- Hardware parameters are DERIVED from the RAPID hardware config (clock,
  SM count, tensor FLOP rate, L2 and DRAM size/bandwidth), so there is a
  single source of truth for device parameters. The RAPID config does not
  describe per-SM resources (registers, shared-memory capacity, warp
  slots), so occupancy is pinned at one resident CTA per SM, and published
  A100-class defaults are used for the memory/MMA latencies and the
  SMEM-bandwidth ratio (see the module constants below).
- The model runs with ideal utilizations and zero fixed overhead; its
  latency already embeds shape-dependent efficiency. RAPID's
  tech_param.core.util is applied by the caller as a residual global scale
  (latency / util), and RAPID's own kernel_launch_overhead is added by the
  caller exactly as for the native backend (no double counting).
- bf16/fp16 (2-byte) precision only; other precisions fall back to the
  native tile model. FlashAttention paths always stay native.
- Kernel selection ranks the CTA-tile catalog with a cheap heuristic,
  prices the shortlist with the full timeline, and picks the lowest
  latency; a small energy estimate breaks exact latency ties.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional

# The backend only prices 2-byte (bf16/fp16) GEMMs.
_BF16_BYTES = 2.0

# Defaults for device details the RAPID hardware config does not describe
# (A100-class published/calibrated values).
_SECTOR_SIZE_BYTES = 32  # L2 transaction granularity
_SMEM_LATENCY_CYCLES = 29.0
_L2_LATENCY_CYCLES = 261.5
_DRAM_LATENCY_CYCLES = 466.3
# Aggregate shared-memory bandwidth, expressed as a multiple of DRAM
# bandwidth (no SMEM level in the RAPID memory hierarchy).
_SMEM_BW_HBM_MULTIPLIER = 8.0
# Tensor-core MMA instruction latency by mma (m, n, k) shape.
_MMA_LATENCY_CYCLES = {(16, 8, 8): 17.5, (16, 8, 16): 26.0}
_MMA_LATENCY_DEFAULT_CYCLES = 8.0
# Without register/shared-memory/warp-slot capacities every occupancy limit
# defaults to one CTA, so exactly one CTA is resident per SM.
_RESIDENT_CTAS_PER_SM = 1

# How many heuristically-ranked kernel candidates get the full timeline.
_SELECTION_SHORTLIST_SIZE = 12

# Energy constants (J/flop, J/byte, W). Latency is the selection objective;
# these only break exact latency ties between kernel candidates.
_TENSOR_ENERGY_J_PER_FLOP = 1.3e-12
_L2_ENERGY_J_PER_BYTE = 3.8e-11
_HBM_ENERGY_J_PER_BYTE = 1.05e-10
_STATIC_POWER_W = 1.0


@dataclass(frozen=True)
class GpuSpec:
    """Device parameters derived from the RAPID hardware config."""

    clock_hz: float
    num_sms: int
    tensor_flops_per_s: float  # bf16/fp16 tensor-core rate, device total
    l2_size_bytes: int
    l2_bandwidth_bytes_per_s: float
    hbm_size_bytes: int
    hbm_bandwidth_bytes_per_s: float


@dataclass(frozen=True)
class GemmProblemSpec:
    """A[m,k] x B[k,n] (per batch; weights are per-batch, beta is zero)."""

    batch: int
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class GemmKernelSpec:
    """CTA-tiled GEMM kernel shape (CUTLASS sm80-style template)."""

    name: str
    cta_m: int
    cta_n: int
    cta_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    mma_m: int
    mma_n: int
    mma_k: int
    pipeline_stages: int
    warps_per_cta: int
    num_warp_tile_k: int = 1
    slice_k: bool = False


# sm80 bf16/fp16 kernel catalog (cta MxNxK / warp MxNxK / warps / stages).
# Catalog order matters: it is the final tie-break in kernel selection.
_GEMM_KERNEL_CATALOG = (
    GemmKernelSpec("sm80_256x128x32_64x64x32_8w3s", 256, 128, 32, 64, 64, 32, 16, 8, 16, 3, 8),
    GemmKernelSpec("sm80_128x256x32_64x64x32_8w3s", 128, 256, 32, 64, 64, 32, 16, 8, 16, 3, 8),
    GemmKernelSpec("sm80_128x128x32_64x64x32_4w3s", 128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 4),
    GemmKernelSpec("sm80_128x64x32_64x32x32_4w3s", 128, 64, 32, 64, 32, 32, 16, 8, 16, 3, 4),
    GemmKernelSpec("sm80_64x128x32_32x64x32_4w3s", 64, 128, 32, 32, 64, 32, 16, 8, 16, 3, 4),
    GemmKernelSpec("sm80_64x64x32_32x32x32_4w3s", 64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 4),
    GemmKernelSpec("sm80_256x64x32_64x32x32_8w3s", 256, 64, 32, 64, 32, 32, 16, 8, 16, 3, 8),
    GemmKernelSpec("sm80_64x256x32_32x64x32_8w3s", 64, 256, 32, 32, 64, 32, 16, 8, 16, 3, 8),
    GemmKernelSpec("sm80_128x32x32_64x32x32_2w3s", 128, 32, 32, 64, 32, 32, 16, 8, 16, 3, 2),
    GemmKernelSpec("sm80_32x128x32_32x64x32_2w3s", 32, 128, 32, 32, 64, 32, 16, 8, 16, 3, 2),
    GemmKernelSpec("sm80_64x32x32_32x32x32_2w3s", 64, 32, 32, 32, 32, 32, 16, 8, 16, 3, 2),
    GemmKernelSpec("sm80_32x64x32_32x32x32_2w3s", 32, 64, 32, 32, 32, 32, 16, 8, 16, 3, 2),
)


@dataclass(frozen=True)
class GridAccounting:
    """CTA grid shape plus useful vs issued (tile-padded) FLOPs."""

    blocks_m: int
    blocks_n: int
    k_stages: int
    cta_count: int
    useful_flops: float
    issued_flops: float
    tile_efficiency: float


@dataclass(frozen=True)
class TrafficAccounting:
    """Per-level byte traffic (sector-rounded transactions).

    A tiles are re-fetched from L2 once per column of CTAs and B tiles once
    per row (first-touch L2 reuse); DRAM sees each tile once. The epilogue
    never reads C (beta is zero), so reads are A+B only.
    """

    a_l2_requested_bytes: int
    b_l2_requested_bytes: int
    a_dram_unique_bytes: int
    b_dram_unique_bytes: int
    d_store_transaction_bytes: int
    l2_requested_bytes: int
    dram_unique_bytes: int
    smem_read_bytes: int
    smem_write_bytes: int

    @property
    def smem_total_bytes(self) -> int:
        return self.smem_read_bytes + self.smem_write_bytes


@dataclass(frozen=True)
class OccupancyResult:
    """Wave decomposition of the CTA grid over the SMs."""

    num_sms: int
    resident_ctas_per_sm: int
    ctas_per_wave: int
    wave_count: int
    full_wave_ctas: int
    last_wave_ctas: int
    last_wave_busy_sms: int
    last_wave_lazy_sms: int
    last_wave_busy_ctas_per_sm: int
    last_wave_lazy_ctas_per_sm: int
    tail_efficiency: float


@dataclass(frozen=True)
class WavePipelineResult:
    """Cycle timeline of one CTA wave (prologue / k-loop work / epilogue)."""

    active_ctas: int
    busy_sms: int
    lazy_sms: int
    busy_ctas_per_sm: int
    lazy_ctas_per_sm: int
    start_cycles: float
    work_cycles: float
    end_cycles: float
    total_cycles: float
    sm_stage_cycles: float
    sm_last_stage_cycles: float
    smem_group_cycles: float
    math_group_cycles: float
    math_issue_group_cycles: float
    math_latency_group_cycles: float
    memory_full_stage_cycles: float
    memory_last_stage_cycles: float
    l2_full_stage_cycles: float
    l2_last_stage_cycles: float
    dram_full_stage_cycles: float
    dram_last_stage_cycles: float
    epilogue_global_cycles: float
    epilogue_smem_cycles: float
    slice_k_extra_cycles: float
    epilogue_smem_bytes: float
    epilogue_l2_bytes: float
    epilogue_dram_bytes: float
    l2_epilogue_cycles: float
    dram_epilogue_cycles: float
    exposed_l2_cycles: float
    exposed_dram_cycles: float


@dataclass(frozen=True)
class TimelineResult:
    """Whole-kernel timeline: full waves plus the (partial) last wave."""

    kernel_cycles: float
    cta_cycles: float
    full_wave: WavePipelineResult
    last_wave: WavePipelineResult
    prologue_cycles: float
    work_cycles: float
    stage_cycles: float
    epilogue_cycles: float


@dataclass(frozen=True)
class GemmCandidateEvaluation:
    """One catalog kernel priced with the full timeline."""

    kernel: GemmKernelSpec
    latency_s: float
    selection_energy_j: float
    tile_efficiency: float
    cheap_rank: int


class ExtendedRooflineGemmBackend:
    """Per-TimeCalculation backend instance (caches the derived GpuSpec)."""

    def __init__(self, hw_config: Any) -> None:
        core = hw_config.tech_config.core
        freq = float(core.operating_frequency)
        num_sms = int(core.num_bundles)
        tensor_flops = (
            freq
            * num_sms
            * float(core.num_mcu_per_bundle)
            * float(core.nominal_flop_rate_per_mcu)
        )
        dram = hw_config.tech_config.DRAM
        l2 = hw_config.tech_config.SRAML2
        self._gpu = GpuSpec(
            clock_hz=freq,
            num_sms=num_sms,
            tensor_flops_per_s=tensor_flops,
            l2_size_bytes=int(l2.size),
            l2_bandwidth_bytes_per_s=float(l2.bandwidth),
            hbm_size_bytes=int(dram.size),
            hbm_bandwidth_bytes_per_s=float(dram.bandwidth),
        )
        self._warned_precision = False

    def gemm_latency_s(
        self, m: int, k: int, n: int, precision_bytes: float, batch: int = 1
    ) -> Optional[float]:
        """Kernel latency for A[m,k] x B[k,n] (batched if batch > 1).

        Returns None when the backend cannot model the request (caller falls
        back to the native tile model).
        """
        if abs(float(precision_bytes) - 2.0) > 1e-6:
            if not self._warned_precision:
                warnings.warn(
                    "extended_roofline GEMM backend supports 2-byte precisions "
                    "only; falling back to the native tile model.",
                    stacklevel=2,
                )
                self._warned_precision = True
            return None
        m, k, n, batch = int(m), int(k), int(n), max(1, int(batch))
        if min(m, k, n) < 1:
            return None
        # Skinny GEMV regime (decode): the extended roofline models CTA-tiled
        # GEMM kernels; real decode kernels are weight-streaming GEMV/split-K
        # and run near the DRAM roofline, which the native model captures
        # better. Defer to the native path below the smallest CTA tile extent.
        if min(m, n) < 128:
            return None
        problem = GemmProblemSpec(batch=batch, m=m, n=n, k=k)
        selected = _select_gemm_kernel(problem, self._gpu)
        latency = float(selected.latency_s)
        if latency <= 0.0 or latency != latency:
            return None
        return latency


def create(hw_config: Any) -> ExtendedRooflineGemmBackend:
    """Create the extended-roofline GEMM backend."""
    return ExtendedRooflineGemmBackend(hw_config)


def _select_gemm_kernel(
    problem: GemmProblemSpec, gpu: GpuSpec
) -> GemmCandidateEvaluation:
    """Rank the catalog cheaply, price the shortlist, pick the best kernel."""
    ranked = sorted(
        _GEMM_KERNEL_CATALOG,
        key=lambda kernel: _cheap_kernel_score(kernel, problem, gpu),
    )
    candidates = []
    for rank, kernel in enumerate(ranked[:_SELECTION_SHORTLIST_SIZE], start=1):
        candidates.append(_evaluate_gemm_kernel(problem, kernel, gpu, rank))
    return min(
        candidates,
        key=lambda candidate: (
            candidate.latency_s,
            candidate.selection_energy_j,
            -candidate.tile_efficiency,
            candidate.cheap_rank,
        ),
    )


def _evaluate_gemm_kernel(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    gpu: GpuSpec,
    cheap_rank: int,
) -> GemmCandidateEvaluation:
    grid = _grid_accounting(problem, kernel)
    traffic = _traffic_accounting(problem, kernel, grid)
    occupancy = _occupancy(grid, gpu.num_sms)
    timeline = _timeline(problem, kernel, grid, traffic, occupancy, gpu)
    latency_s = timeline.kernel_cycles / gpu.clock_hz
    return GemmCandidateEvaluation(
        kernel=kernel,
        latency_s=latency_s,
        selection_energy_j=_selection_energy_j(grid, traffic, latency_s),
        tile_efficiency=grid.tile_efficiency,
        cheap_rank=cheap_rank,
    )


def _selection_energy_j(
    grid: GridAccounting, traffic: TrafficAccounting, latency_s: float
) -> float:
    """Kernel energy estimate, charging tile-padding waste; tie-break only."""
    compute_j = grid.useful_flops * _TENSOR_ENERGY_J_PER_FLOP
    hbm_j = traffic.dram_unique_bytes * _HBM_ENERGY_J_PER_BYTE
    l2_j = traffic.l2_requested_bytes * _L2_ENERGY_J_PER_BYTE
    static_j = _STATIC_POWER_W * latency_s
    energy_j = compute_j + hbm_j + l2_j + static_j
    extra_flops = max(0.0, grid.issued_flops - grid.useful_flops)
    return energy_j + extra_flops * _TENSOR_ENERGY_J_PER_FLOP


def _cheap_kernel_score(
    kernel: GemmKernelSpec, problem: GemmProblemSpec, gpu: GpuSpec
) -> float:
    """Heuristic pre-ranking of catalog kernels (lower is better)."""
    grid = _grid_accounting(problem, kernel)
    occupancy = _occupancy(grid, gpu.num_sms)
    problem_aspect = problem.m / max(problem.n, 1)
    tile_aspect = kernel.cta_m / max(kernel.cta_n, 1)
    aspect_penalty = abs(math.log2(problem_aspect / tile_aspect))
    wave_penalty = 1.0 - occupancy.tail_efficiency
    underfill_penalty = max(0, occupancy.ctas_per_wave - grid.cta_count) / max(
        occupancy.ctas_per_wave, 1
    )
    k_penalty = 0.2 if problem.k < kernel.cta_k else 0.0
    tile_efficiency_penalty = 1.0 - grid.tile_efficiency
    return (
        4.0 * tile_efficiency_penalty
        + 1.2 * underfill_penalty
        + 0.7 * wave_penalty
        + 0.35 * aspect_penalty
        + k_penalty
    )


def _grid_accounting(problem: GemmProblemSpec, kernel: GemmKernelSpec) -> GridAccounting:
    blocks_m = _ceil_div(problem.m, kernel.cta_m)
    blocks_n = _ceil_div(problem.n, kernel.cta_n)
    k_stages = _ceil_div(problem.k, kernel.cta_k)
    cta_count = problem.batch * blocks_m * blocks_n
    useful_flops = float(2 * problem.batch * problem.m * problem.n * problem.k)
    issued_flops = float(
        2
        * problem.batch
        * blocks_m
        * blocks_n
        * k_stages
        * kernel.cta_m
        * kernel.cta_n
        * kernel.cta_k
    )
    return GridAccounting(
        blocks_m=blocks_m,
        blocks_n=blocks_n,
        k_stages=k_stages,
        cta_count=cta_count,
        useful_flops=useful_flops,
        issued_flops=issued_flops,
        tile_efficiency=useful_flops / issued_flops if issued_flops else 0.0,
    )


def _smem_load_per_cta_elements(
    problem: GemmProblemSpec, kernel: GemmKernelSpec, grid: GridAccounting
) -> int:
    """SMEM->register load work per CTA over the full K-loop (elements)."""
    warp_m_tiles = _ceil_div(kernel.cta_m, kernel.warp_m)
    warp_n_tiles = _ceil_div(kernel.cta_n, kernel.warp_n)
    warp_k_tiles = _ceil_div(kernel.cta_k, kernel.warp_k)
    return (
        (kernel.warp_m + kernel.warp_n)
        * warp_m_tiles
        * warp_n_tiles
        * grid.k_stages
        * warp_k_tiles
        * kernel.warp_k
    )


def _shared_memory_bytes_per_cta(
    problem: GemmProblemSpec, kernel: GemmKernelSpec, grid: GridAccounting
) -> int:
    return _ceil_scalar_bytes(
        _smem_load_per_cta_elements(problem, kernel, grid) * _BF16_BYTES
    )


def _traffic_accounting(
    problem: GemmProblemSpec, kernel: GemmKernelSpec, grid: GridAccounting
) -> TrafficAccounting:
    dtype_bytes = _BF16_BYTES
    sector_size = _SECTOR_SIZE_BYTES

    m_tiles = _tile_lengths(problem.m, kernel.cta_m)
    n_tiles = _tile_lengths(problem.n, kernel.cta_n)
    k_tiles = _tile_lengths(problem.k, kernel.cta_k)

    a_unique_tx = sum(
        _sector_round_bytes(m_len * k_len * dtype_bytes, sector_size)
        for m_len in m_tiles
        for k_len in k_tiles
    )
    b_unique_one_batch_tx = sum(
        _sector_round_bytes(k_len * n_len * dtype_bytes, sector_size)
        for n_len in n_tiles
        for k_len in k_tiles
    )
    a_requested_tx = problem.batch * grid.blocks_n * a_unique_tx
    b_requested_tx = problem.batch * grid.blocks_m * b_unique_one_batch_tx
    a_unique_dram_tx = problem.batch * a_unique_tx
    b_unique_dram_tx = problem.batch * b_unique_one_batch_tx

    d_store_tx = problem.batch * sum(
        _sector_round_bytes(m_len * n_len * dtype_bytes, sector_size)
        for m_len in m_tiles
        for n_len in n_tiles
    )
    l2_read = a_requested_tx + b_requested_tx
    l2_write = d_store_tx
    hbm_read = a_unique_dram_tx + b_unique_dram_tx
    hbm_write = d_store_tx

    stage_operand_bytes = _ceil_scalar_bytes(
        (kernel.cta_m * kernel.cta_k + kernel.cta_k * kernel.cta_n) * dtype_bytes
    )
    smem_write = grid.cta_count * grid.k_stages * stage_operand_bytes
    smem_read = grid.cta_count * _shared_memory_bytes_per_cta(problem, kernel, grid)

    return TrafficAccounting(
        a_l2_requested_bytes=a_requested_tx,
        b_l2_requested_bytes=b_requested_tx,
        a_dram_unique_bytes=a_unique_dram_tx,
        b_dram_unique_bytes=b_unique_dram_tx,
        d_store_transaction_bytes=d_store_tx,
        l2_requested_bytes=l2_read + l2_write,
        dram_unique_bytes=hbm_read + hbm_write,
        smem_read_bytes=smem_read,
        smem_write_bytes=smem_write,
    )


def _occupancy(grid: GridAccounting, num_sms: int) -> OccupancyResult:
    resident = _RESIDENT_CTAS_PER_SM
    ctas_per_wave = num_sms * resident
    wave_count = max(1, _ceil_div(grid.cta_count, ctas_per_wave))
    final_wave_ctas = grid.cta_count - (wave_count - 1) * ctas_per_wave
    tail_efficiency = final_wave_ctas / ctas_per_wave if ctas_per_wave else 1.0
    lazy_ctas_per_sm, busy_remainder = divmod(final_wave_ctas, num_sms)
    busy_ctas_per_sm = lazy_ctas_per_sm + (1 if busy_remainder else 0)
    busy_sms = busy_remainder if busy_remainder else (num_sms if final_wave_ctas else 0)
    lazy_sms = num_sms - busy_sms
    return OccupancyResult(
        num_sms=num_sms,
        resident_ctas_per_sm=resident,
        ctas_per_wave=ctas_per_wave,
        wave_count=wave_count,
        full_wave_ctas=ctas_per_wave,
        last_wave_ctas=final_wave_ctas,
        last_wave_busy_sms=busy_sms,
        last_wave_lazy_sms=lazy_sms,
        last_wave_busy_ctas_per_sm=busy_ctas_per_sm,
        last_wave_lazy_ctas_per_sm=lazy_ctas_per_sm,
        tail_efficiency=tail_efficiency,
    )


def _timeline(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    occupancy: OccupancyResult,
    gpu: GpuSpec,
) -> TimelineResult:
    clock_hz = gpu.clock_hz
    peak_compute_flops_per_cycle = gpu.tensor_flops_per_s / clock_hz
    peak_hbm_bw_per_cycle = gpu.hbm_bandwidth_bytes_per_s / clock_hz
    peak_l2_bw_per_cycle = gpu.l2_bandwidth_bytes_per_s / clock_hz
    peak_smem_bw_per_cycle = (
        gpu.hbm_bandwidth_bytes_per_s * _SMEM_BW_HBM_MULTIPLIER / clock_hz
    )

    shared_latency = _SMEM_LATENCY_CYCLES
    l2_latency = _L2_LATENCY_CYCLES
    dram_latency = _DRAM_LATENCY_CYCLES
    tensor_latency = _MMA_LATENCY_CYCLES.get(
        (kernel.mma_m, kernel.mma_n, kernel.mma_k), _MMA_LATENCY_DEFAULT_CYCLES
    )

    groups_k = _k_groups(kernel.cta_k, kernel)
    last_stage_k = (problem.k - 1) % kernel.cta_k + 1 if problem.k else 0
    last_stage_groups_k = _k_groups(last_stage_k, kernel)
    memory_pipeline_groups = _ceil_div(grid.k_stages, kernel.pipeline_stages)
    last_memory_pipeline_stages = (grid.k_stages - 1) % kernel.pipeline_stages + 1

    load_stage_count = max(1, grid.cta_count * grid.k_stages)
    avg_l2_load_bytes_per_cta_stage = (
        traffic.a_l2_requested_bytes + traffic.b_l2_requested_bytes
    ) / load_stage_count
    avg_dram_load_bytes_per_cta_stage = (
        traffic.a_dram_unique_bytes + traffic.b_dram_unique_bytes
    ) / load_stage_count

    full_wave = _wave_pipeline(
        problem=problem,
        kernel=kernel,
        grid=grid,
        traffic=traffic,
        active_ctas=occupancy.full_wave_ctas,
        busy_sms=occupancy.num_sms,
        lazy_sms=0,
        busy_ctas_per_sm=occupancy.resident_ctas_per_sm,
        lazy_ctas_per_sm=occupancy.resident_ctas_per_sm,
        peak_compute_flops_per_cycle=peak_compute_flops_per_cycle,
        peak_smem_bw_per_cycle=peak_smem_bw_per_cycle,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        shared_latency_cycles=shared_latency,
        tensor_latency_cycles=tensor_latency,
        l2_latency_cycles=l2_latency,
        dram_latency_cycles=dram_latency,
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
    )
    last_wave = _wave_pipeline(
        problem=problem,
        kernel=kernel,
        grid=grid,
        traffic=traffic,
        active_ctas=occupancy.last_wave_ctas,
        busy_sms=occupancy.last_wave_busy_sms,
        lazy_sms=occupancy.last_wave_lazy_sms,
        busy_ctas_per_sm=occupancy.last_wave_busy_ctas_per_sm,
        lazy_ctas_per_sm=occupancy.last_wave_lazy_ctas_per_sm,
        peak_compute_flops_per_cycle=peak_compute_flops_per_cycle,
        peak_smem_bw_per_cycle=peak_smem_bw_per_cycle,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        shared_latency_cycles=shared_latency,
        tensor_latency_cycles=tensor_latency,
        l2_latency_cycles=l2_latency,
        dram_latency_cycles=dram_latency,
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
    )

    full_wave_count = max(0, occupancy.wave_count - 1)
    kernel_cycles = full_wave.total_cycles * full_wave_count + last_wave.total_cycles
    prologue_cycles = full_wave.start_cycles * full_wave_count + last_wave.start_cycles
    work_cycles = full_wave.work_cycles * full_wave_count + last_wave.work_cycles
    epilogue_cycles = full_wave.end_cycles * full_wave_count + last_wave.end_cycles
    cta_cycles = full_wave.total_cycles / max(1, occupancy.resident_ctas_per_sm)
    stage_cycles = full_wave.work_cycles / max(1, grid.k_stages)

    return TimelineResult(
        kernel_cycles=kernel_cycles,
        cta_cycles=cta_cycles,
        full_wave=full_wave,
        last_wave=last_wave,
        prologue_cycles=prologue_cycles,
        work_cycles=work_cycles,
        stage_cycles=stage_cycles,
        epilogue_cycles=epilogue_cycles,
    )


def _wave_pipeline(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    active_ctas: int,
    busy_sms: int,
    lazy_sms: int,
    busy_ctas_per_sm: int,
    lazy_ctas_per_sm: int,
    peak_compute_flops_per_cycle: float,
    peak_smem_bw_per_cycle: float,
    peak_l2_bw_per_cycle: float,
    peak_hbm_bw_per_cycle: float,
    shared_latency_cycles: float,
    tensor_latency_cycles: float,
    l2_latency_cycles: float,
    dram_latency_cycles: float,
    avg_l2_load_bytes_per_cta_stage: float,
    avg_dram_load_bytes_per_cta_stage: float,
    groups_k: int,
    last_stage_groups_k: int,
    last_stage_k: int,
    memory_pipeline_groups: int,
    last_memory_pipeline_stages: int,
) -> WavePipelineResult:
    if active_ctas <= 0:
        return WavePipelineResult(
            active_ctas=0,
            busy_sms=0,
            lazy_sms=busy_sms + lazy_sms,
            busy_ctas_per_sm=0,
            lazy_ctas_per_sm=0,
            start_cycles=0.0,
            work_cycles=0.0,
            end_cycles=0.0,
            total_cycles=0.0,
            sm_stage_cycles=0.0,
            sm_last_stage_cycles=0.0,
            smem_group_cycles=0.0,
            math_group_cycles=0.0,
            math_issue_group_cycles=0.0,
            math_latency_group_cycles=0.0,
            memory_full_stage_cycles=0.0,
            memory_last_stage_cycles=0.0,
            l2_full_stage_cycles=0.0,
            l2_last_stage_cycles=0.0,
            dram_full_stage_cycles=0.0,
            dram_last_stage_cycles=0.0,
            epilogue_global_cycles=0.0,
            epilogue_smem_cycles=0.0,
            slice_k_extra_cycles=0.0,
            epilogue_smem_bytes=0.0,
            epilogue_l2_bytes=0.0,
            epilogue_dram_bytes=0.0,
            l2_epilogue_cycles=0.0,
            dram_epilogue_cycles=0.0,
            exposed_l2_cycles=0.0,
            exposed_dram_cycles=0.0,
        )

    total_sms = max(1, busy_sms + lazy_sms)
    per_sm_smem_bw = peak_smem_bw_per_cycle / total_sms
    per_smsp_compute = peak_compute_flops_per_cycle / total_sms / 4.0
    active_warps = max(0, busy_ctas_per_sm) * kernel.warps_per_cta

    effective_warp_m = min(kernel.warp_m, kernel.cta_m)
    effective_warp_n = min(kernel.warp_n, kernel.cta_n)
    effective_warp_k = _effective_warp_k(kernel)
    warptile_smem_to_reg_bytes = (
        (effective_warp_m + effective_warp_n) * effective_warp_k * _BF16_BYTES
    )
    smem_group_cycles = (
        max(active_warps * warptile_smem_to_reg_bytes / max(per_sm_smem_bw, 1.0e-12),
            shared_latency_cycles)
        if active_warps > 0
        else 0.0
    )

    smsp_warps = _ceil_div(active_warps, 4) if active_warps else 0
    concurrent_mma = (
        smsp_warps
        * _ceil_div(effective_warp_m, kernel.mma_m)
        * _ceil_div(effective_warp_n, kernel.mma_n)
    )
    mma_k_iters = max(1, _ceil_div(effective_warp_k, kernel.mma_k))
    math_issue_group_cycles = (
        mma_k_iters
        * concurrent_mma
        * (2 * kernel.mma_m * kernel.mma_n * kernel.mma_k)
        / max(per_smsp_compute, 1.0e-12)
        if concurrent_mma > 0
        else 0.0
    )
    math_latency_group_cycles = mma_k_iters * tensor_latency_cycles if concurrent_mma > 0 else 0.0
    math_group_cycles = max(math_issue_group_cycles, math_latency_group_cycles)
    per_group_cycles = max(smem_group_cycles, math_group_cycles)
    if kernel.pipeline_stages >= 2:
        # Register fragment double-buffering overlaps the SMEM->RF load of the
        # next warp-tile group with the MMA of the current one, and the
        # cp.async multi-stage pipeline extends this across k-stages, so the
        # steady-state stage runs at the slower of the two rates.
        sm_stage_cycles = max(1, groups_k) * per_group_cycles
        sm_last_stage_cycles = max(1, last_stage_groups_k) * per_group_cycles
    else:
        sm_stage_cycles = smem_group_cycles + math_group_cycles + (
            max(0, groups_k - 1) * per_group_cycles
        )
        sm_last_stage_cycles = smem_group_cycles + math_group_cycles + (
            max(0, last_stage_groups_k - 1) * per_group_cycles
        )

    full_memory_stages = min(kernel.pipeline_stages, grid.k_stages)
    last_stage_fraction = last_stage_k / max(1, kernel.cta_k)
    last_memory_stage_equiv = max(
        last_stage_fraction,
        float(max(0, last_memory_pipeline_stages - 1)) + last_stage_fraction,
    )
    (
        memory_full_stage_cycles,
        l2_full_stage_cycles,
        dram_full_stage_cycles,
    ) = _memory_pipeline_stage_cycles(
        active_ctas=active_ctas,
        stage_equivalent=float(full_memory_stages),
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        l2_latency_cycles=l2_latency_cycles,
        dram_latency_cycles=dram_latency_cycles,
    )
    (
        memory_last_stage_cycles,
        l2_last_stage_cycles,
        dram_last_stage_cycles,
    ) = _memory_pipeline_stage_cycles(
        active_ctas=active_ctas,
        stage_equivalent=last_memory_stage_equiv,
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        l2_latency_cycles=l2_latency_cycles,
        dram_latency_cycles=dram_latency_cycles,
    )

    sm_path_cycles = (
        sm_stage_cycles * max(0, grid.k_stages - 1) + sm_last_stage_cycles
        if grid.k_stages > 1
        else sm_last_stage_cycles
    )
    memory_path_cycles = sm_stage_cycles + sm_last_stage_cycles + (
        memory_full_stage_cycles * max(0, memory_pipeline_groups - 1)
        + memory_last_stage_cycles
    )
    work_cycles = max(sm_path_cycles, memory_path_cycles) if grid.k_stages > 1 else sm_path_cycles
    start_cycles = (
        memory_full_stage_cycles
        if grid.k_stages >= kernel.pipeline_stages
        else memory_last_stage_cycles
    )

    wave_fraction = active_ctas / max(1, grid.cta_count)
    l2_epilogue_bytes = (
        wave_fraction * traffic.d_store_transaction_bytes
        if peak_l2_bw_per_cycle > 0.0
        else 0.0
    )
    dram_epilogue_bytes = wave_fraction * traffic.d_store_transaction_bytes
    warptile_reg_to_smem_bytes = effective_warp_m * effective_warp_n * _BF16_BYTES
    epilogue_smem_cycles = (
        max(active_warps * warptile_reg_to_smem_bytes / max(per_sm_smem_bw, 1.0e-12),
            shared_latency_cycles)
        if active_warps > 0
        else 0.0
    )
    slice_k_ld_cycles = epilogue_smem_cycles if kernel.slice_k else 0.0
    slice_k_st_cycles = (
        max(
            active_warps
            * warptile_reg_to_smem_bytes
            / max(per_sm_smem_bw * kernel.num_warp_tile_k, 1.0e-12),
            shared_latency_cycles,
        )
        if kernel.slice_k and active_warps > 0
        else 0.0
    )
    slice_k_extra_cycles = slice_k_ld_cycles + slice_k_st_cycles
    epilogue_smem_bytes = wave_fraction * traffic.d_store_transaction_bytes
    if kernel.slice_k:
        epilogue_smem_bytes += wave_fraction * traffic.d_store_transaction_bytes * (
            1.0 + 1.0 / max(1, kernel.num_warp_tile_k)
        )
    l2_epilogue_cycles = _service_cycles(
        l2_epilogue_bytes, peak_l2_bw_per_cycle, l2_latency_cycles
    )
    dram_epilogue_cycles = _service_cycles(
        dram_epilogue_bytes, peak_hbm_bw_per_cycle, dram_latency_cycles
    )
    epilogue_global_cycles = max(l2_epilogue_cycles, dram_epilogue_cycles)
    end_cycles = epilogue_smem_cycles + slice_k_extra_cycles + epilogue_global_cycles

    l2_pipeline_cycles = (
        l2_full_stage_cycles * max(0, memory_pipeline_groups - 1) + l2_last_stage_cycles
    )
    dram_pipeline_cycles = (
        dram_full_stage_cycles * max(0, memory_pipeline_groups - 1) + dram_last_stage_cycles
    )
    exposed_l2_cycles = max(0.0, l2_pipeline_cycles - sm_path_cycles)
    exposed_dram_cycles = max(0.0, dram_pipeline_cycles - sm_path_cycles)

    return WavePipelineResult(
        active_ctas=active_ctas,
        busy_sms=busy_sms,
        lazy_sms=lazy_sms,
        busy_ctas_per_sm=busy_ctas_per_sm,
        lazy_ctas_per_sm=lazy_ctas_per_sm,
        start_cycles=start_cycles,
        work_cycles=work_cycles,
        end_cycles=end_cycles,
        total_cycles=start_cycles + work_cycles + end_cycles,
        sm_stage_cycles=sm_stage_cycles,
        sm_last_stage_cycles=sm_last_stage_cycles,
        smem_group_cycles=smem_group_cycles,
        math_group_cycles=math_group_cycles,
        math_issue_group_cycles=math_issue_group_cycles,
        math_latency_group_cycles=math_latency_group_cycles,
        memory_full_stage_cycles=memory_full_stage_cycles,
        memory_last_stage_cycles=memory_last_stage_cycles,
        l2_full_stage_cycles=l2_full_stage_cycles,
        l2_last_stage_cycles=l2_last_stage_cycles,
        dram_full_stage_cycles=dram_full_stage_cycles,
        dram_last_stage_cycles=dram_last_stage_cycles,
        epilogue_global_cycles=epilogue_global_cycles,
        epilogue_smem_cycles=epilogue_smem_cycles,
        slice_k_extra_cycles=slice_k_extra_cycles,
        epilogue_smem_bytes=epilogue_smem_bytes,
        epilogue_l2_bytes=l2_epilogue_bytes,
        epilogue_dram_bytes=dram_epilogue_bytes,
        l2_epilogue_cycles=l2_epilogue_cycles,
        dram_epilogue_cycles=dram_epilogue_cycles,
        exposed_l2_cycles=exposed_l2_cycles,
        exposed_dram_cycles=exposed_dram_cycles,
    )


def _memory_pipeline_stage_cycles(
    *,
    active_ctas: int,
    stage_equivalent: float,
    avg_l2_load_bytes_per_cta_stage: float,
    avg_dram_load_bytes_per_cta_stage: float,
    peak_l2_bw_per_cycle: float,
    peak_hbm_bw_per_cycle: float,
    l2_latency_cycles: float,
    dram_latency_cycles: float,
) -> tuple:
    l2_bytes = active_ctas * stage_equivalent * avg_l2_load_bytes_per_cta_stage
    dram_bytes = active_ctas * stage_equivalent * avg_dram_load_bytes_per_cta_stage
    l2_cycles = _service_cycles(l2_bytes, peak_l2_bw_per_cycle, l2_latency_cycles)
    dram_cycles = _service_cycles(dram_bytes, peak_hbm_bw_per_cycle, dram_latency_cycles)
    return max(l2_cycles, dram_cycles), l2_cycles, dram_cycles


def _service_cycles(bytes_count: float, bandwidth_bytes_per_cycle: float, latency_cycles: float) -> float:
    if bytes_count <= 0.0 or bandwidth_bytes_per_cycle <= 0.0:
        return 0.0
    return max(bytes_count / bandwidth_bytes_per_cycle, latency_cycles)


def _effective_warp_k(kernel: GemmKernelSpec) -> int:
    if kernel.slice_k:
        return max(1, _ceil_div(kernel.cta_k, kernel.num_warp_tile_k))
    return min(kernel.warp_k, kernel.cta_k)


def _k_groups(k_extent: int, kernel: GemmKernelSpec) -> int:
    if k_extent <= 0:
        return 0
    k_per_warp_tile = _ceil_div(k_extent, kernel.num_warp_tile_k)
    return max(1, _ceil_div(k_per_warp_tile, kernel.warp_k))


def _tile_lengths(total: int, tile: int) -> tuple:
    full, rem = divmod(total, tile)
    values = [tile] * full
    if rem:
        values.append(rem)
    return tuple(values) or (0,)


def _sector_round_bytes(value: float, sector_size: int) -> int:
    scalar = _ceil_scalar_bytes(value)
    if scalar == 0:
        return 0
    return _ceil_div(scalar, sector_size) * sector_size


def _ceil_scalar_bytes(value: float) -> int:
    return int(math.ceil(value))


def _ceil_div(numerator, denominator) -> int:
    return int(math.ceil(numerator / denominator))
