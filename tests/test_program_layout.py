# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``program.layout.RankLayout`` (migration stage M0).

Pure stdlib tests: fake network layouts are built with ``SimpleNamespace``
objects mimicking ``config.py``'s ``NetworkDimensionLayout`` (attrs
``parallelisms`` / ``size`` / ``label`` / ``optimize_2dmap`` /
``topology_type`` / ``size_2d``).

Run: ./.venv/bin/python -m pytest tests/test_program_layout.py -q
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from program.layout import CANONICAL_AXES, RankLayout, cluster_coords


# ---------------------------------------------------------------------------
# Fake network-layout builders
# ---------------------------------------------------------------------------


def _dim(
    parallelisms,
    size,
    *,
    label="dim",
    optimize_2dmap=False,
    topology_type="Ring",
    size_2d=None,
):
    return SimpleNamespace(
        id=label,
        label=label,
        size=size,
        topology_type=topology_type,
        parallelisms=tuple(parallelisms),
        optimize_2dmap=optimize_2dmap,
        size_2d=size_2d,
    )


def _network(*dims):
    return SimpleNamespace(dimensions=list(dims))


def _sizes(tp=1, cp=1, ep=1, pp=1, dp=1):
    return {"tp": tp, "cp": cp, "ep": ep, "pp": pp, "dp": dp}


def _build(network, sizes, enforce=False, **kwargs):
    layout, cfg = RankLayout.from_network_layout(network, sizes, enforce, **kwargs)
    return layout, cfg


# ---------------------------------------------------------------------------
# linearize / coords_of round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout",
    [
        RankLayout(("tp",), {"tp": 4}, {"tp": 1}),
        RankLayout(("dp",), {"dp": 5}, {"dp": 1}),
        RankLayout(("tp", "pp"), {"tp": 3, "pp": 4}, {"tp": 1, "pp": 3}),
        RankLayout(
            ("tp", "cp", "pp", "dp"),
            {"tp": 2, "cp": 2, "ep": 1, "pp": 2, "dp": 2},
            {"tp": 1, "cp": 2, "pp": 4, "dp": 8},
        ),
        RankLayout(
            CANONICAL_AXES,
            {"tp": 2, "cp": 2, "ep": 2, "pp": 2, "dp": 2},
            {"tp": 1, "cp": 2, "ep": 4, "pp": 8, "dp": 16},
        ),
    ],
    ids=["tp-only", "dp-only", "tp-pp", "four-axis", "all-five"],
)
def test_linearize_coords_of_round_trip(layout):
    num = layout.num_ranks()
    seen = set()
    for rank in range(num):
        coords = layout.coords_of(rank)
        assert set(coords) == set(layout.axis_order)
        for axis, coord in coords.items():
            assert 0 <= coord < layout.axis_sizes[axis]
        assert layout.linearize(coords) == rank
        seen.add(tuple(sorted(coords.items())))
    assert len(seen) == num  # bijective


def test_linearize_missing_axes_default_to_zero():
    layout = RankLayout(("tp", "pp"), {"tp": 2, "pp": 3}, {"tp": 1, "pp": 2})
    assert layout.linearize({}) == 0
    assert layout.linearize({"pp": 2}) == 4
    assert layout.linearize({"tp": 1}) == 1


def test_linearize_out_of_range_coordinate_message():
    layout = RankLayout(("tp",), {"tp": 2}, {"tp": 1})
    with pytest.raises(ValueError, match=re.escape("Coordinate 2 for axis 'tp' is out of range <2")):
        layout.linearize({"tp": 2})
    with pytest.raises(ValueError, match=re.escape("Coordinate -1 for axis 'tp' is out of range <2")):
        layout.linearize({"tp": -1})


def test_linearize_missing_stride_keyerror_message():
    # Legacy _configure_rank_layout accepted descriptors with partial stride
    # dicts and only failed at linearize time; RankLayout keeps that contract.
    layout = RankLayout(("tp", "cp"), {"tp": 2, "cp": 2}, {"tp": 1})
    with pytest.raises(KeyError, match=re.escape("Rank layout stride missing for axis 'cp'")):
        layout.linearize({"tp": 0, "cp": 1})


def test_coords_of_out_of_range_rank():
    layout = RankLayout(("tp",), {"tp": 4}, {"tp": 1})
    with pytest.raises(ValueError):
        layout.coords_of(4)
    with pytest.raises(ValueError):
        layout.coords_of(-1)


# ---------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------


def _full_layout():
    network = _network(
        _dim(("tp", "cp"), 4, label="nvlink"),
        _dim(("pp",), 2, label="pp_dim"),
        _dim(("dp",), 2, label="dp_dim"),
    )
    layout, _ = _build(network, _sizes(tp=2, cp=2, pp=2, dp=2))
    return layout


def test_subset_transformer_axes():
    sub = _full_layout().subset(["tp", "cp", "ep"])
    assert sub.axis_order == ("tp", "cp")  # ep inactive -> not in axis_order
    assert sub.axis_sizes == {"tp": 2, "cp": 2}  # filtered to subset axes only
    assert sub.axis_strides == {"tp": 1, "cp": 2}  # strides restart from 1
    assert sub.num_ranks() == 4


def test_subset_pipeline_axes():
    sub = _full_layout().subset(["pp", "dp"])
    assert sub.axis_order == ("pp", "dp")
    assert sub.axis_sizes == {"pp": 2, "dp": 2}
    assert sub.axis_strides == {"pp": 1, "dp": 2}
    assert sub.descriptor() == {
        "axis_order": ["pp", "dp"],
        "axis_sizes": {"pp": 2, "dp": 2},
        "axis_strides": {"pp": 1, "dp": 2},
        "stage_span": 4,
    }


def test_subset_preserves_canonical_order_regardless_of_allowed_order():
    sub = _full_layout().subset(["dp", "pp"])
    assert sub.axis_order == ("pp", "dp")


def test_subset_empty_when_axes_absent():
    sub = _full_layout().subset(["ep"])
    assert sub.axis_order == ()
    assert sub.descriptor() == {
        "axis_order": [],
        "axis_sizes": {},
        "axis_strides": {},
        "stage_span": 1,
    }


def test_subset_round_trip_consistency():
    full = _full_layout()
    sub = full.subset(["pp", "dp"])
    for rank in range(sub.num_ranks()):
        assert sub.linearize(sub.coords_of(rank)) == rank


# ---------------------------------------------------------------------------
# descriptor: exact legacy dict shape
# ---------------------------------------------------------------------------


def test_descriptor_matches_hand_built_legacy_descriptor():
    # Hand-built expectation replicating LLMExecutionDispatcher.
    # _build_rank_layout_descriptor for dims [tp,cp]x4, [pp]x2, [dp]x2:
    # axis_sizes keeps ALL FIVE canonical axes (even inactive ep), axis_order
    # only the axes present in the network layout, strides row-major (tp
    # fastest), stage_span the full product.
    layout, optimize_cfg = _build(
        _network(
            _dim(("tp", "cp"), 4, label="nvlink"),
            _dim(("pp",), 2, label="pp_dim"),
            _dim(("dp",), 2, label="dp_dim"),
        ),
        _sizes(tp=2, cp=2, pp=2, dp=2),
    )
    assert optimize_cfg is None
    descriptor = layout.descriptor()
    assert descriptor == {
        "axis_order": ["tp", "cp", "pp", "dp"],
        "axis_sizes": {"tp": 2, "cp": 2, "ep": 1, "pp": 2, "dp": 2},
        "axis_strides": {"tp": 1, "cp": 2, "pp": 4, "dp": 8},
        "stage_span": 16,
    }
    # Legacy shape details: axis_order is a *list*, and axis_sizes preserves
    # the tp,cp,ep,pp,dp insertion order (serialization-visible).
    assert isinstance(descriptor["axis_order"], list)
    assert list(descriptor["axis_sizes"]) == ["tp", "cp", "ep", "pp", "dp"]
    assert list(descriptor["axis_strides"]) == ["tp", "cp", "pp", "dp"]


def test_descriptor_canonicalizes_axis_order():
    # dp-major declaration order in the network layout must still yield
    # canonical tp,...,dp axis_order (legacy reorder step).
    layout, _ = _build(
        _network(_dim(("dp",), 2, label="outer"), _dim(("tp",), 2, label="inner")),
        _sizes(tp=2, dp=2),
    )
    assert layout.descriptor()["axis_order"] == ["tp", "dp"]
    assert layout.descriptor()["axis_strides"] == {"tp": 1, "dp": 2}


def test_descriptor_empty_axis_order():
    layout, _ = _build(_network(_dim((), 1, label="empty")), _sizes())
    assert layout.descriptor() == {
        "axis_order": [],
        "axis_sizes": {"tp": 1, "cp": 1, "ep": 1, "pp": 1, "dp": 1},
        "axis_strides": {},
        "stage_span": 1,
    }


# ---------------------------------------------------------------------------
# from_network_layout: validation errors (exact legacy messages)
# ---------------------------------------------------------------------------


def test_unsupported_axis_error_astrasim_context():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unsupported parallelism axis 'foo' in network layout. "
            "Supported axes for AstraSim integration are: tp, cp, ep, pp, dp."
        ),
    ):
        _build(_network(_dim(("foo",), 2)), _sizes())


def test_unsupported_axis_error_memory_estimation_context():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unsupported parallelism axis 'foo' in network layout. "
            "Supported axes for memory estimation are: tp, cp, ep, pp, dp."
        ),
    ):
        _build(
            _network(_dim(("foo",), 2)),
            _sizes(),
            unsupported_axis_context="memory estimation",
        )


def test_dimension_size_mismatch_error():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Network dimension 'nvlink' size mismatch: declared 4, but parallelism factors imply 2."
        ),
    ):
        _build(_network(_dim(("tp",), 4, label="nvlink")), _sizes(tp=2))


@pytest.mark.parametrize(
    "axis_kwargs, message",
    [
        ({"tp": 2}, "Network layout must include 'tp' when tensor parallelism > 1."),
        ({"cp": 2}, "Network layout must include 'cp' when context parallelism > 1."),
        ({"ep": 2}, "Network layout must include 'ep' when expert parallelism > 1."),
        ({"pp": 2}, "Network layout must include 'pp' when pipeline parallelism > 1."),
    ],
)
def test_active_axis_must_appear_in_layout(axis_kwargs, message):
    network = _network(_dim(("dp",), 2, label="dp_dim"))
    with pytest.raises(ValueError, match=re.escape(message)):
        _build(network, _sizes(dp=2, **axis_kwargs))


def test_active_dp_without_dp_axis_is_allowed():
    # Legacy quirk: dp > 1 never requires a 'dp' axis in the network layout.
    layout, _ = _build(_network(_dim(("tp",), 2, label="nvlink")), _sizes(tp=2, dp=4))
    assert layout.axis_order == ("tp",)
    assert layout.descriptor()["stage_span"] == 2


# ---------------------------------------------------------------------------
# from_network_layout: optimize_2dmap extraction
# ---------------------------------------------------------------------------


def test_optimize_2dmap_extraction_success():
    layout, cfg = _build(
        _network(
            _dim(("tp",), 4, label="mesh", optimize_2dmap=True, topology_type="Mesh2D", size_2d=(2, 2)),
        ),
        _sizes(tp=4),
    )
    assert layout.axis_order == ("tp",)
    assert cfg == {
        "dimension_index": 0,
        "topology": "Mesh2D",
        "size": 4,
        "parallelisms": ("tp",),
        "dims": (2, 2),
    }


def test_optimize_2dmap_extraction_without_size_2d():
    _, cfg = _build(
        _network(_dim(("tp",), 4, optimize_2dmap=True, topology_type="Mesh2D")),
        _sizes(tp=4),
    )
    assert cfg == {
        "dimension_index": 0,
        "topology": "Mesh2D",
        "size": 4,
        "parallelisms": ("tp",),
    }


def test_optimize_2dmap_only_first_dimension():
    with pytest.raises(
        ValueError,
        match=re.escape("optimize_2dmap is only supported on the first network dimension."),
    ):
        _build(
            _network(
                _dim(("tp",), 2),
                _dim(("dp",), 2, optimize_2dmap=True, topology_type="Mesh2D"),
            ),
            _sizes(tp=2, dp=2),
        )


def test_optimize_2dmap_requires_topology_type():
    with pytest.raises(
        ValueError,
        match=re.escape("optimize_2dmap requires a topology type on the target dimension."),
    ):
        _build(
            _network(_dim(("tp",), 2, optimize_2dmap=True, topology_type=None)),
            _sizes(tp=2),
        )


def test_optimize_2dmap_requires_explicit_size():
    with pytest.raises(
        ValueError,
        match=re.escape("optimize_2dmap requires an explicit dimension size."),
    ):
        _build(
            _network(_dim(("tp",), None, optimize_2dmap=True, topology_type="Mesh2D")),
            _sizes(tp=2),
        )


def test_optimize_2dmap_extraction_can_be_disabled():
    # Memory estimation path never performed the optimize_2dmap scan.
    _, cfg = _build(
        _network(_dim(("tp",), 4, optimize_2dmap=True, topology_type="Mesh2D")),
        _sizes(tp=4),
        extract_optimize_2dmap=False,
    )
    assert cfg is None


# ---------------------------------------------------------------------------
# from_network_layout: hierarchical/hybrid cluster enforcement
# ---------------------------------------------------------------------------


def test_enforce_cluster_rejects_sched_axes_before_cluster():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Dimension 'sched' carries ['pp'] while the cluster axes ['tp'] are not yet fully mapped "
            "(covered so far: [])."
        ),
    ):
        _build(
            _network(_dim(("pp",), 2, label="sched"), _dim(("tp",), 2, label="nvlink")),
            _sizes(tp=2, pp=2),
            enforce=True,
        )


def test_enforce_cluster_rejects_incomplete_cluster_coverage():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "must jointly carry the active TP/CP/EP axes ['cp', 'tp'] (found only ['tp'])."
        ),
    ):
        _build(
            _network(_dim(("tp",), 2, label="nvlink")),
            _sizes(tp=2, cp=2),
            enforce=True,
        )


def test_enforce_cluster_accepts_leading_cluster_dims():
    layout, _ = _build(
        _network(
            _dim(("tp", "cp"), 4, label="nvlink"),
            _dim(("ep",), 2, label="internode"),
            _dim(("dp",), 2, label="dp_dim"),
        ),
        _sizes(tp=2, cp=2, ep=2, dp=2),
        enforce=True,
    )
    assert layout.axis_order == ("tp", "cp", "ep", "dp")


def test_no_enforcement_when_flag_false():
    layout, _ = _build(
        _network(_dim(("pp",), 2, label="sched"), _dim(("tp",), 2, label="nvlink")),
        _sizes(tp=2, pp=2),
        enforce=False,
    )
    assert layout.axis_order == ("tp", "pp")


# ---------------------------------------------------------------------------
# from_network_layout: memory-estimation empty-axis-order early out
# ---------------------------------------------------------------------------


def test_empty_axis_order_returns_none_when_requested():
    layout, cfg = _build(
        _network(_dim((), 1, label="empty")),
        _sizes(),
        extract_optimize_2dmap=False,
        empty_axis_order_is_none=True,
    )
    assert layout is None
    assert cfg is None


def test_empty_axis_order_skips_active_axis_checks_when_none_requested():
    # Legacy memory estimation returned None *before* the active-axis checks;
    # the dispatcher path (flag off) raises instead.
    network = _network(_dim((), 1, label="empty"))
    layout, _ = _build(network, _sizes(tp=2), empty_axis_order_is_none=True)
    assert layout is None
    with pytest.raises(ValueError, match=re.escape("Network layout must include 'tp'")):
        _build(network, _sizes(tp=2))


# ---------------------------------------------------------------------------
# cluster_coords: legacy hw-id decomposition
# ---------------------------------------------------------------------------


def test_cluster_coords_matches_legacy_hw_id_formula():
    # Layout tp=2, cp=2, pp=2 (no dp): legacy hw_id == stage * par_degree + rank.
    layout, _ = _build(
        _network(_dim(("tp", "cp"), 4, label="nvlink"), _dim(("pp",), 2, label="pp_dim")),
        _sizes(tp=2, cp=2, pp=2),
    )
    for stage in range(2):
        for rank in range(4):
            coords = cluster_coords(
                layout.axis_order, rank, stage, tp_size=2, cp_size=2, ep_size=1, pp_size=2
            )
            assert coords == {"tp": rank % 2, "cp": (rank // 2) % 2, "pp": stage}
            assert layout.linearize(coords) == stage * 4 + rank


def test_cluster_coords_only_sets_axes_in_order():
    coords = cluster_coords(("tp",), 3, 0, tp_size=4, cp_size=1, ep_size=1, pp_size=1)
    assert coords == {"tp": 3}


def test_cluster_coords_ep_decomposition():
    coords = cluster_coords(
        ("tp", "cp", "ep"), 7, 0, tp_size=2, cp_size=2, ep_size=2, pp_size=1
    )
    assert coords == {"tp": 1, "cp": 1, "ep": 1}


def test_cluster_coords_pp_bounds_check_message():
    with pytest.raises(ValueError, match=re.escape("stage_id 5 is out of range for pp=2")):
        cluster_coords(("pp",), 0, 5, tp_size=1, cp_size=1, ep_size=1, pp_size=2)
    with pytest.raises(ValueError, match=re.escape("stage_id -1 is out of range for pp=2")):
        cluster_coords(("pp",), 0, -1, tp_size=1, cp_size=1, ep_size=1, pp_size=2)
