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

"""``CommunicatorFactory`` — communicator groups CONSTRUCTED, never inferred
(INTERFACES.md §3.3, L2).

This module replaces three pieces of legacy machinery:

* ``legacy_lowering._compute_stage_axis_coords`` (:135-155) — re-derives a
  device's axis coordinates by *dividing the flat device id* by the axis
  sizes. :meth:`RankLayout.coords_of` is the one implementation.
* ``legacy_lowering._build_axis_groups`` (:157-181) — precomputes, per axis, a
  ``(dp_idx, other-axis coordinates) -> member list`` table. Here a group is
  computed on demand from the anchor device, which is the same partition.
* ``legacy_lowering._assign_collective_labels_with_members`` (:196-282) — in
  particular the composite hack at :243-248::

      if axis == "ep":
          if tp_size > 1 and participants == tp_size * ep_size:
              composite_axes = ("tp", "ep")

  i.e. the ``("tp","ep")`` communicator of a ``tp_ep``-routed MoE all-to-all
  was RECOVERED from the analytical participant count. Here it is an ordinary
  input: ``members(("tp","ep"), anchor)`` is one call with no special case, and
  the axis tuple is DECLARED upstream on ``CommSpec.axes`` (which the MoE
  routing policy fills from ``MoERoutingPolicy.routing_axes()``).

Two invariants this module is responsible for (INTERFACES §3.5):

* **P2** — every ``GroupKey.members`` equals
  ``CommunicatorFactory.members(declared_axes, device)``; no group is ever
  derived from a participant count. ``CommunicatorFactory`` never receives a
  participant count: :meth:`CommunicatorFactory.members` takes axes and an
  anchor, and nothing else. (The heuristic is quoted above for provenance;
  the only executable occurrence left in ``program/`` is
  ``legacy_lowering.py:247``, which P5 deletes.)
* an axis whose size is 1 (or which the layout does not carry at all)
  contributes a **singleton** group — the emitter's zero-duration ``*_noop``
  COMP substitution (``et_emit.py:313-320``) still applies unchanged
  (BUG_LEDGER Class B 10g).

``dp`` never reaches this module — and that is now ENFORCED, not merely intended
(INTERFACES §3.3 amendment, 2026-07-26). ``GroupKey.members`` are pre-DP devices
and dp replication is stamped at emission (``ir.py:93-104``), so

* the layout this factory is constructed over is ``Placement.group_layout``,
  which is the device layout MINUS ``dp``; and
* :meth:`CommunicatorFactory._spanned_axes` **raises** :class:`GroupError` on
  ``"dp"``, because with dp absent from the layout the "absent axis -> singleton"
  rule above would otherwise silently delete every dp reducer (a singleton group
  becomes a zero-duration ``*_noop`` at emission). A dp requirement carries
  ``group=None, is_dp=True`` instead — see
  :meth:`program.work.SyncRequirement.is_dp` and :meth:`groups_for`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from program.ir import GroupKey
from program.layout import CANONICAL_AXES, RankLayout
from program.types import DP_AXIS, AxisName, DeviceId


class GroupError(ValueError):
    """A communicator group was requested for axes the layout cannot express."""


def canonical_axis_label(axes: Sequence[AxisName]) -> str:
    """Diagnostics label for an axis tuple: canonical order, ``+``-joined.

    ``("ep","tp")`` and ``("tp","ep")`` produce the same label, so the label is
    a function of the axis SET. Membership remains the identity — nothing in
    ``program/`` dispatches on this string.
    """
    seen = {str(axis) for axis in axes}
    ordered = [axis for axis in CANONICAL_AXES if axis in seen]
    ordered.extend(sorted(axis for axis in seen if axis not in CANONICAL_AXES))
    if not ordered:
        raise GroupError("A communicator needs at least one axis")
    return "+".join(ordered)


@dataclass(frozen=True)
class CommunicatorFactory:
    """Builds communicator member sets from a :class:`RankLayout`.

    ``layout`` is the device space the group members live in — i.e. the layout
    a :class:`program.placement.Placement` exposes for its granularity, so
    members are always device ids of the same program.
    """

    layout: RankLayout

    # -- membership --------------------------------------------------------
    def members(self, axes: Sequence[AxisName], anchor: DeviceId) -> Tuple[DeviceId, ...]:
        """Every device sharing ``anchor``'s coordinates on all axes NOT in
        ``axes``, ranging over the full product of ``axes``. Sorted ascending.

        Composite axes are ORDINARY input: ``("tp","ep")`` is one call, not a
        special case (this is the deleted ``legacy_lowering.py:243-248`` hack).

        An axis the layout does not carry (absent from ``axis_order``, or of
        size 1) contributes exactly one coordinate, so a communicator over
        such an axis is a singleton — matching legacy, where ``_build_axis_groups``
        skipped ``size <= 1`` axes and the caller fell back to ``[anchor]``.
        """
        spanned = self._spanned_axes(axes)
        base = self.layout.coords_of(int(anchor))
        if not spanned:
            return (DeviceId(int(anchor)),)

        ranges = [range(self.layout.axis_sizes[axis]) for axis in spanned]
        found: List[int] = []
        for combo in itertools.product(*ranges):
            coords: Dict[AxisName, int] = dict(base)
            coords.update(zip(spanned, combo))
            found.append(self.layout.linearize(coords))
        return tuple(DeviceId(rank) for rank in sorted(set(found)))

    def group_for(self, axes: Sequence[AxisName], anchor: DeviceId) -> GroupKey:
        """``GroupKey(axis=canonical label, members=members(axes, anchor))``."""
        return GroupKey(
            axis=canonical_axis_label(axes),
            members=self.members(axes, anchor),
        )

    def groups_for(
        self, req: "SyncRequirement", placement: "Placement"  # noqa: F821
    ) -> Tuple[Optional[GroupKey], ...]:
        """One entry per instance device of ``req``: a :class:`GroupKey`, or
        ``None`` when ``req.is_dp``.

        ``req`` is an L1 ``SyncRequirement`` (``.axes`` + ``.spread`` +
        ``.place_on``); ``placement`` resolves the spread to instance devices
        (:meth:`program.placement.Placement.devices_for_sync`). Kept here, next
        to :meth:`members`, so the only place a communicator is built is this
        class.

        AMENDMENT to INTERFACES §3.3 (dated 2026-07-26): the return type is
        ``Tuple[Optional[GroupKey], ...]``. A dp requirement gets ``None`` — it
        maps to ``CollectiveOp(group=None, is_dp=True)`` and its wire members
        are stamped at emission (``ir.py:93-104``). Before this, a dp
        requirement got a group whose members were not devices at all.
        """
        devices = placement.devices_for_sync(req)
        if req.is_dp:
            return tuple(None for _ in devices)
        return tuple(self.group_for(req.axes, device) for device in devices)

    # -- introspection -----------------------------------------------------
    def spans(self, axes: Sequence[AxisName]) -> int:
        """Number of members any group over ``axes`` has (the product of the
        spanned axis sizes). 1 means every such group is a singleton."""
        span = 1
        for axis in self._spanned_axes(axes):
            span *= self.layout.axis_sizes[axis]
        return span

    def partition(self, axes: Sequence[AxisName]) -> Tuple[Tuple[DeviceId, ...], ...]:
        """Every distinct group over ``axes``, in ascending first-member order.

        Used by ``validate.py`` V7 ("a grouped collective is instantiated on
        every member device of its group") and by the placement tests.
        """
        seen: Dict[Tuple[DeviceId, ...], None] = {}
        for device in range(self.layout.num_ranks()):
            seen.setdefault(self.members(axes, DeviceId(device)), None)
        return tuple(sorted(seen, key=lambda group: group[0]))

    # -- internals ---------------------------------------------------------
    def _spanned_axes(self, axes: Sequence[AxisName]) -> Tuple[AxisName, ...]:
        if not axes:
            raise GroupError("A communicator needs at least one axis")
        spanned: List[AxisName] = []
        for axis in axes:
            name = str(axis)
            if name == DP_AXIS:
                # A dp collective has no GroupKey at all: its members are not
                # devices of this layout. Silently returning a singleton (which
                # et_emit turns into a zero-duration *_noop) would DELETE the
                # reducer; returning members that vary the dp coordinate would
                # produce ids outside Placement.devices(). Both were live before
                # this raise existed.
                raise GroupError(
                    "'dp' is not a communicator axis of a device layout: dp "
                    "replication is stamped at emission over pre-dp device ids "
                    "(ir.py:93-104). A dp requirement carries group=None, "
                    "is_dp=True (SyncRequirement.is_dp)."
                )
            if name not in CANONICAL_AXES:
                raise GroupError(
                    f"Unknown communicator axis {name!r} (canonical axes: {CANONICAL_AXES})"
                )
            if name not in self.layout.axis_order:
                # The layout does not resolve this axis -> singleton contribution.
                continue
            if self.layout.axis_sizes[name] <= 1:
                continue
            if name in spanned:
                raise GroupError(f"Duplicate communicator axis {name!r} in {tuple(axes)!r}")
            spanned.append(name)
        # Canonical order keeps the itertools.product deterministic; the result
        # is sorted anyway, so this only pins the iteration.
        return tuple(axis for axis in CANONICAL_AXES if axis in spanned)


def dedupe_groups(groups: Iterable[GroupKey]) -> Tuple[GroupKey, ...]:
    """Distinct groups in first-seen order (membership is the identity)."""
    seen: Dict[Tuple[str, Tuple[DeviceId, ...]], GroupKey] = {}
    for group in groups:
        seen.setdefault((group.axis, group.members), group)
    return tuple(seen.values())


__all__ = [
    "CommunicatorFactory",
    "GroupError",
    "canonical_axis_label",
    "dedupe_groups",
]
