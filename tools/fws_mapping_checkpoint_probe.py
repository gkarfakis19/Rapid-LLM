"""Checkpoint diagnostics; never changes the production mapper.

The flexible-phase probe preserves every weight's macro/slot placement and
allows an occupied slot to fire at a phase other than its physical address.
It tests the extra D39 slot=phase restriction, not a production replacement.
"""
import argparse
import json
import math
import sys
from pathlib import Path


def flexible_phase(mapping, period):
    p = mapping.platform
    occupied = {m.index: set() for m in mapping.macros}
    events = []
    previous_end = None
    previous_chip = None
    for b in mapping.bundles:
        earliest = 0 if previous_end is None else previous_end + p.hop_gap_ticks
        if previous_chip is not None and previous_chip != b.chip:
            earliest += p.seam_ticks
        times = []
        for q in b.placements:
            candidates = [r for r in range(period) if r not in occupied[q.macro]]
            assert candidates, 'period must fit all resident firings on each macro'
            tick = min(earliest + ((r - earliest) % period) for r in candidates)
            occupied[q.macro].add(tick % period)
            times.append(tick)
            events.append((b.spec.order, q.macro, q.slot, tick, b.chip))
        previous_end, previous_chip = max(times), b.chip

    # Independent explicit event check for eight overlapping tokens. Exactly
    # one slot-layer fires per macro/tick, so all active ADCs share one input.
    resources = set()
    identities = set()
    expected = {(q.macro, q.slot) for b in mapping.bundles for q in b.placements}
    for token in range(8):
        seen = set()
        previous = None
        for b in mapping.bundles:
            be = [e for e in events if e[0] == b.spec.order]
            start = min(e[3] for e in be) + token * period
            end = max(e[3] for e in be) + token * period
            if previous is not None:
                gap = p.hop_gap_ticks + (p.seam_ticks if previous[1] != b.chip else 0)
                assert start >= previous[0] + gap
            for _, macro, slot, tick, _ in be:
                assert (macro, tick + token * period) not in resources
                resources.add((macro, tick + token * period))
                assert (token, macro, slot) not in identities
                identities.add((token, macro, slot))
                seen.add((macro, slot))
            previous = (end, b.chip)
        assert seen == expected
    latency = max(e[3] for e in events) + 1
    return {'period_ticks': period, 'L_ticks': latency,
            'D': math.ceil(latency / period),
            'tokens_per_s': 1e9 / (period * p.tick_ns),
            'checked_tokens': 8,
            'events': [{'bundle_order': b, 'macro': m, 'physical_slot': s,
                        'tick': t, 'chip': c} for b, m, s, t, c in events]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    sys.path[:0] = [str(args.repo), str(args.repo / 'tools')]
    import fws_slot_mapper as sm
    import fws_toy_slotmap as toy
    raw, spec, bundles = toy.load_config(args.repo / 'configs/toy/toy_slotmap.yaml')
    rows, sensitivity, relaxed = [], [], []
    for point in raw['points']:
        p = sm.SlotPlatform(spec['rows'], point['z'], point['y'], spec['chip_macros'])
        m = sm.map_chain(p, bundles)
        led = sm.ledgers(m)
        cells_allocated = len(m.macros) * p.rows * p.y
        cells_useful = sum(b.columns for b in bundles) * p.rows
        row = dict(id=point['id'], mux=p.mux, **led)
        row['all_unused_capacity_pct'] = 100 * (1 - cells_useful / cells_allocated)
        row['unassigned_slot_count'] = sum(len(mm.free) for mm in m.macros)
        row['filled_macro_capacity_fraction'] = cells_useful / (led['macros_used'] * p.rows * p.y)
        rows.append(row)
        period = max(len(mm.placed) for mm in m.macros)
        if period < p.mux:
            candidates = [flexible_phase(m, pp) for pp in range(period, p.mux)]
            chosen = [('maximum_throughput', candidates[0])]
            same_d = [cc for cc in candidates if cc['D'] <= led['D']]
            if same_d and same_d[0]['period_ticks'] != period:
                chosen.append(('no_extra_resident_sequences', same_d[0]))
            for label, r in chosen:
                r.update(id=point['id'], variant=label, original_period_ticks=p.mux,
                         original_L_ticks=led['L_ticks'], original_D=led['D'],
                         same_placement=True,
                         throughput_ratio=p.mux / r['period_ticks'])
                relaxed.append(r)
        for chip_macros in [6, 12, 24, 48]:
            for hop in [2, 8, 32]:
                pp = sm.SlotPlatform(p.rows, p.z, p.y, chip_macros,
                                     hop_gap_ticks=hop)
                mm = sm.map_chain(pp, bundles)
                ll = sm.ledgers(mm)
                doc = sm.to_document(mm, title='sensitivity', authored='2026-09-18',
                                     author='checkpoint diagnostic', command='probe_mapping.py',
                                     layers=spec['layers'])
                assert sm.check_document(doc)['ok']
                sensitivity.append(dict(id=point['id'], chip_macros=chip_macros,
                                        hop_gap_ticks=hop, **ll))
    result = {'baseline': rows, 'sensitivity': sensitivity,
              'flexible_phase_counterexamples': relaxed,
              'scope': 'Toy only. Flexible-phase rows relax D39 slot=phase, preserve '
                       'physical weight placement, and check eight tokens under '
                       'the same simplified dependencies and provisioned bandwidth.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for r in rows:
        print('%s mux=%d L=%d D=%d reported_waste=%.2f%% unused_capacity=%.2f%% ADC_duty=%.2f%%'
              % (r['id'], r['mux'], r['L_ticks'], r['D'], 100*r['waste_frac'],
                 r['all_unused_capacity_pct'], 100*r['macro_duty_mean']))
    for r in relaxed:
        print('FLEXIBLE %s period %d->%d L %d->%d D %d->%d speedup=%.3fx (8 tokens checked)'
              % (r['id'], r['original_period_ticks'], r['period_ticks'],
                 r['original_L_ticks'], r['L_ticks'], r['original_D'], r['D'],
                 r['throughput_ratio']))


if __name__ == '__main__':
    main()
