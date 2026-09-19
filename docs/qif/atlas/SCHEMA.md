# `fws_atlas/1` — the mapping export contract

The System Atlas (P5) consumes exactly one file. This document defines it.
It is frozen at phase 0, before renderer work, so P3 (placement) and P4
(time) have one target. `atlas.html` is the reference implementation of the
loader; `fixture_llama7b_tp2.json` is the reference document.

Two rules govern everything below.

- **The atlas computes only to refuse, never to display.** Every number on
  screen arrives in the file. The loader does arithmetic in one place only:
  the conformance checks, where a computed value is compared against a
  declared one and a disagreement stops the drawing. No computed value is
  ever painted. This is D21's "one accounting per metric" applied to a
  viewer (AUDIT finding 4).
- **A mapping that cannot be drawn is not a mapping.** Chips are enumerated
  objects, macros name their owners, tile columns are real spans. A producer
  that emits a quotient instead of a placement fails the loader by
  construction (AUDIT finding 1, D21).

## Top level

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema` | string | yes | Must be `"fws_atlas/1"`. |
| `title` | string | yes | One line naming the mapping. |
| `subtitle` | string | no | One line of context. |
| `provenance` | object | yes | Where the numbers came from. See below. |
| `cards` | object | yes | Device-card id → card. P2 owns the vocabulary (D14). |
| `systems[]` | array | yes | One entry, or two under PD disaggregation (D16). |
| `chips[]` | array | yes | Every chip, as an object. Never a count. |
| `macros[]` | array | yes | Every macro, as an object (A3). |
| `tiles[]` | array | yes | Every tile, as an object (A3). |
| `links[]` | array | yes | Traffic-bearing links only (D17). |
| `metrics[]` | array | yes | Labeled headline figures. |
| `relaxations[]` | array | yes | Disclosed relaxations; may be empty. |
| `folding` | object | no | The P7.6 folding card's contract, `fws_fold/1`. See "The P7.6 folding extensions". |
| `passes` | object | no | The P7.6 macro time-lane view's contract, `fws_pass/1`. See the same section. |

`chips[]`, `macros[]`, `tiles[]` and `links[]` are flat tables joined by id.
Containment is stated twice — `chips[].macros` holds macro ids and
`macros[].chip` holds the chip id — and the loader checks both directions.
The redundancy is the point: it is what makes a broken export visible.

Ids are strings and are the sort key for every layout decision. Same file,
same picture, every time.

## `provenance`

| Field | Type | Meaning |
|---|---|---|
| `authored` | string | ISO date. |
| `author` | string | Who or what produced the file. |
| `reference_run` | object | `command`, `report`, `parallelism` — the run the grounded numbers come from. |
| `grounded[]` | string[] | Prose list of what the run supplied. |
| `invented_fields[]` | string[] | Dotted paths, `macros[].duty_cycle` style. Every value at a listed path is marked in the UI. |
| `note` | string | Anything a reader must know before quoting a number. |

### The `fixture_invented` convention

A fixture is allowed to carry numbers no tool produced — that is what makes
it useful to code against — but those numbers must never leave the file
wearing the clothes of a result. Two marks, one meaning:

- `"fixture_invented": true` on any object, or an array of field names
  (`"fixture_invented": ["duty_cycle"]`) to mark only some of its fields.
- `provenance.invented_fields[]` applies the same mark by dotted path to
  every matching object, so a 700-macro fixture does not repeat itself.

The renderer draws a `◇` next to every marked value, prints the paths in the
legend, and shows the count in the header. A marked number is a placeholder
of the right shape, never evidence.

## `cards`

Device cards, keyed by id (D14; ADJ-4 for the shared digital chiplet card).

| Field | Type | Meaning |
|---|---|---|
| `label` | string | Human name. |
| `kind` | string | `analog_macro` \| `digital_engine`. |
| `capacity` | object | `rows`, `cols_adc`, `mux`, `stored_cols`. `stored_cols` = `cols_adc × mux` (P2 semantics). |
| `area_mm2` | number \| null | `null` renders as the unknown hatch. The atlas never fills a gap with a guess. |
| `bits_per_cell` | int \| null | Slicing is present iff `bits_per_cell < weight_bits` (ADJ-4). |
| `weight_bits` | int | Stored weight width. |
| `slicing` | bool | Must agree with the `bits_per_cell` test. |
| `basis` | string | Where the card's numbers came from. |

Accuracy is not a card field and is not a field anywhere in this schema
(D23).

## `systems[]`

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Referenced by `chips[].system`. |
| `role` | string | `unified` \| `prefill` \| `decode`. Two systems render side by side under PD disaggregation (D16). |
| `phase` | string | The operating point the metrics and duty cycles describe: `prefill` \| `decode`. A `unified` system still has one. |
| `label` | string | Human name. |
| `parallelism` | object | `tp`, `ep`, `pp` degrees. |
| `models[]` | array | The owner vocabulary (P1). |

`models[]` entries carry `id`, `label`, the architecture facts the tile cards
print (`layers`, `hidden_dim`, `intermediate_size`, `vocab_size`,
`attention`, `precision`), `blocks[]` — the legal `owner.block` values — and
`endpoint_blocks[]`, the subset that has no layer index. `attention` is
`{type, num_heads, head_dim}`. `serving` is optional and carries the serving
point the system card prints: `{batch, prefill_len, decode_len, seq_len}`.

## `chips[]`

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique. |
| `system` | string | Must exist in `systems[]`. |
| `pool` | string | `analog` \| `digital` (D13). Drawn as an outline style, never a color. |
| `label` | string | Human name. |
| `capacity.macro_slots` | int | Positive integer. Must equal `len(macros)`. |
| `groups` | object | `{tp, ep, pp}` indices. |
| `primary_group` | string | `tp` \| `ep` \| `pp` — which axis owns this chip's hue (ADJ-7). |
| `occupancy` | number | Slots holding at least one tile / `macro_slots`. Declared, then checked. |
| `macros[]` | string[] | Macro ids. Every slot is listed, reserved ones included. |

A shared digital chiplet stores no weights, so its `occupancy` is legitimately
`0.0` and its macros carry no tiles. It draws as an unfilled, dashed cell —
which is the correct picture.

## `macros[]`

The atomic resource (A3).

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique. |
| `chip` | string | Must exist, and that chip must list this macro. |
| `card` | string | Must exist in `cards`. |
| `capacity` | object | Optional override; defaults to the card's capacity. |
| `tiles[]` | string[] | Tile ids. May be empty only when `reserved` is set. |
| `occupancy` | number | Stored columns held by tiles / `stored_cols`. Declared, then checked. |
| `duty_cycle` | number \| null | Fraction of the period the macro is active (P4), in `[0, 1]`. `null` means the producer measured none — a placement-only export, or a device P4's timeline does not cover — and draws as the unknown hatch. A `0.0` is a MEASURED idle macro and draws as an empty bar; the two are never the same statement. |
| `digital_pool` | object \| null | The per-macro digital pool D12 requires be reported: `shift_add_units`, `act_lanes`, `area_mm2`. |
| `reserved` | string \| null | Why this macro holds no tiles. Required when `tiles` is empty. |

**The named-owner invariant (D21).** A macro is drawn because it names its
owners. A macro with tiles names them through the tiles; a macro without
tiles names its reservation in `reserved`. A macro with neither is refused —
that is the only way an anonymous resource can be caught.

## `tiles[]`

The allocation unit (A3).

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique. |
| `macro` | string | Must exist, and that macro must list this tile. |
| `owner` | object | `{model, layer, block, expert}`. `model` must exist; `block` must be in that model's `blocks[]`; `layer` is an integer, or `null` only when `block` is an endpoint block. `expert` is `null` for dense models. |
| `shape` | object | `{K, N}` of **this tile**, not of the whole matrix. `K ≤ capacity.rows` and `N == columns.count`. |
| `slice` | object \| null | Bit slicing (D11): `{index, of, bits}`. `null` when the card does not slice. |
| `groups` | object | `{tp, ep, pp}` indices — the color axis (D20). |
| `primary_group` | string | `tp` \| `ep` \| `pp`. Which axis's index picks the hue (ADJ-7). The atlas invents no precedence rule. |
| `columns` | object | `{start, count}` — a real span inside the macro's stored columns. |

**Row partitions.** A weight matrix whose `K` exceeds the macro's rows is
carried by several tiles that share one owner tuple, each with its own `K`.
The schema adds no field for this: the owner tuple identifies the matrix, and
the tiles are its slabs, summed in the per-macro digital pool (D12). The
fixture's `ffn_down` at K = 5504 is the live example.

**Unowned columns are legal** (ADJ-5). Columns no tile spans are simply not
covered; they show as gaps in the column strip and pull `occupancy` down.

## `links[]`

On a fully-connected p2p fabric (D17) the drawn links are the ones that carry
traffic. There is no topology here and no time.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique. |
| `from`, `to` | string | Chip ids. |
| `role` | string | `tp` \| `ep` \| `pp` \| `pd` \| `act` \| `svc`. |
| `bytes` | number | Bytes per `per`. |
| `per` | string | The unit `bytes` is counted over, e.g. `decode_step`, `inference`. |
| `label` | string | What the traffic is. |
| `basis` | string | The arithmetic behind `bytes`. |

`act` extends the plan page's four roles. It carries activation traffic that
is not a parallelism collective: the chip-boundary handoff, and the analog →
shared-digital hop that attention needs (D13). The alternative was to call a
chip boundary `pp`, which ADJ-5 forbids — chip index is not implicitly pp.

`svc` is the SERVICE relationship between a shared digital chiplet and each
analog chip whose act × act work it runs (D13): `from` is the chiplet's chip
id, `to` is the analog chip. Its `bytes` are the traffic that relationship
moves in ONE beat, both directions summed — operands out (Q/K/V, the projected
scan inputs) plus results back — and they are MEASURED off P4's timeline, never
a formula: `basis` prints the two directions, the op count and the per-block
itemization. A direction with 0 B has no lowered transfer in it and `basis`
says which, rather than the total absorbing an estimate.

`svc` rows are PRICED rows. A placement-only export carries none, because P3
has no timeline to measure them on (A1). One `act` link is a chip boundary and
one `svc` link is a wire to the chiplet: a reader who wants the pipeline map's
wire trees reads `svc`, and a reader who wants the pipeline's own handoffs
reads `act`. They are different traffic and are never summed into one row.

Roles are drawn as line dash patterns, never as hue. Hue is the parallelism
group and nothing else (D20).

## `metrics[]`

The header strip. The producer picks the headline; the atlas never does.

| Field | Type | Meaning |
|---|---|---|
| `key` | string | What is being measured. |
| `label` | string | Which figure this is. Non-empty, and unique within a `key`. |
| `value` | number | |
| `unit` | string | |
| `basis` | string | Where the number came from. Required. |
| `system` | string | Optional in a one-system document, **required in kind** in a two-system one: which system this figure measures. A `key` prefixed with the system id (`sys.decode.tokens_per_s`) says the same thing and is what the exporter emits. See rule 28. |
| `fixture_invented` | bool | Optional. |

**Two figures for one metric are legal only when both are labeled.** Fabric
ceiling versus sustained is the live example, and it is exactly the shape of
AUDIT finding 4: two accountings of one metric sitting in one artifact, one
of them quotable out of context. Labels are what make the pair honest, so the
loader requires them.

**In a PD document the rule goes one level up.** Two systems publish the same
LABEL — "steady throughput at the decode terminal (HEADLINE)" — once each,
under different keys, because they are two machines. That is legal and it is
the point of the picture; what is not legal is a reader being unable to tell
whose figure is whose. Rule 28 therefore requires every metric in a
multi-system document to name its system, and the header strip prints the
system's role on every tile.

## `relaxations[]`

`{constraint, value, reason}`, all three non-empty. A non-empty list renders
as a persistent banner, never a footnote (D21, AUDIT finding 5).

## What the loader refuses to draw

On any error the atlas draws nothing and lists every failure in a validation
panel, each naming the field path that failed. Warnings draw, and say so.

| # | Check | Refuses when |
|---|---|---|
| 1 | `schema.id` | `schema` is not `fws_atlas/1`. |
| 2 | `chips[].system` | The system id does not exist. |
| 3 | `macros[].chip` | The chip id does not exist. |
| 4 | `chips[].macros` | A listed macro is missing, or a macro's `chip` back-reference disagrees. |
| 5 | `macros[].tiles` | A listed tile is missing, or a tile's `macro` back-reference disagrees. |
| 6 | `links[].from` / `.to` | The chip id does not exist. |
| 7 | `macros[].card` | The card id is not in `cards`. |
| 8 | `chips[].capacity.macro_slots` | Not a positive integer, or not equal to the number of listed macros. |
| 9 | *derived chip count* | `chips` is not an array, or any field named like a per-chip quotient (`chips_per_…`, `…_per_chip`, `chip_count`) carries a non-integer. A chip count that is a division is not a placement. |
| 10 | `macros[].reserved` | A macro has no tiles and no reservation text. The named-owner invariant. |
| 11 | `tiles[].owner` | `model` unknown, `block` not in the model's `blocks[]`, or `layer` null for a non-endpoint block. |
| 12 | `tiles[].columns` | `start < 0`, `count ≤ 0`, or `start + count > stored_cols`. |
| 13 | `tiles[].columns` | Two tiles on one macro overlap. |
| 14 | `tiles[].shape.N` | `N ≠ columns.count`. |
| 15 | `tiles[].shape.K` | `K > capacity.rows`. |
| 16 | `macros[].occupancy` | Disagrees with the enumerated column spans. |
| 17 | `chips[].occupancy` | Disagrees with the enumerated macros holding tiles. |
| 18 | `metrics[].label` | Two entries share a `key` with a blank or duplicate `label`. |
| 19 | `relaxations[]` | An entry is missing `constraint`, `value` or `reason`. |
| 20 | `tiles[].primary_group` | Not one of `tp`/`ep`/`pp`, or the axis is missing from `groups`. |
| 21 | `chips[].primary_group` | The chip declares a group index a resident tile contradicts on that axis. |
| 22 | *hue capacity* (warning) | More than four distinct `(axis, index)` pairs are co-visible. The excess draws as the over-palette hatch. Hues are never cycled. |
| 23 | *draw budget* (warning) | A chip's enumerated macros would draw more rects than the phase-1 budget of 10000. That chip is not drawn; every other chip is. The count is of RECTS, the unit the budget is stated in, not of macro slots. |
| 24 | `cards.<id>` | `kind` is outside `analog_macro`/`digital_engine`, `weight_bits` is not a positive integer, `capacity.stored_cols ≠ cols_adc × mux`, or `slicing` disagrees with the `bits_per_cell < weight_bits` test (ADJ-4). |
| 25 | `tiles[].slice` | A tile claims a slice role on a card whose `slicing` is false, or the role is not `{index, of, bits}` with `0 ≤ index < of` (D11). |
| 26 | `systems[].role` / `.phase` | `role` is outside `unified`/`prefill`/`decode`, or `phase` is outside `prefill`/`decode` (D16). |
| 27 | `macros[].duty_cycle` | Present and outside `[0, 1]`. It is painted as a bar, so an out-of-range value would render as a plausible full or empty one. |
| 28 | `metrics[].system` | The document holds more than one system and a metric names none of them — no `system` field and no key prefixed with a system id — or it names a system that does not exist. Two inventories publish the same headline label; a figure that does not say whose it is puts AUDIT finding 4 one level up. |
| 29 | `systems[].role` / `links[].role` | A `pd` link with both ends inside one system (a handoff to itself), or a multi-system document whose roles are not exactly one `prefill` and one `decode`, or that carries other than exactly one `pd` link. D16 is two inventories and ONE handoff; two systems with no link between them are two pictures, not a machine. |
| 1 | `title` / `provenance` | The document carries no `title`, no `provenance` object, or a `provenance` without `authored`, `author` and an array `invented_fields[]`. Without provenance every `fixture_invented` mark disappears and a placeholder reads as evidence. |
| 6 | `links[].per` / `.basis` | A byte count that does not say what it is per, or does not say where it came from. |
| 9 | *derived quotient* | The scan covers the document root, `systems[]`, `chips[]`, `macros[]`, `tiles[]`, `links[]` and `metrics[]` — the placement tables included, since that is where a half-ported area-division producer would put per-chip arithmetic. |
| 18 | `metrics[].value` | Not a finite number. The atlas prints a metric verbatim, so a string would be printed as a figure. |

`atlas.html` carries a self-test that corrupts a copy of the loaded document
once per rule and asserts each refusal fires and names the right field. Open
it with `?selftest=1`; there is no button, and it never runs on a plain load —
it clones the document once per rule and that is not work to do on the way to
a picture. It runs 37
cases: the clean document plus one per corruption — one for every rule the
loader can emit, with extra cases for rules 1, 9, 18 and 24, which refuse
more than one way. Every case that indexes a table declares what it NEEDS, so
a document with no link, no relaxation or no tile skips that case instead of
throwing: ViT places no inter-chip link, and the two link cases used to take
the whole results layer down with them. `tests/test_qif_atlas.py` asserts that the integer in this
sentence equals `ATLAS.selfTest(doc).length`, so it cannot rot again. `tests/test_qif_atlas.py` runs the same self-test headlessly in
node, so a rule that stops firing fails the suite rather than a screenshot.

A corruption a given document gives the harness no way to BUILD — rule 13
needs a macro holding two tiles, rule 22 needs a second shared digital
chiplet — is reported as `n/a`, "not constructible on this document". It is
neither a pass nor a miss, and calling it a miss would read as a broken loader
on a perfectly good file. It is counted OUT OF THE DENOMINATOR: the panel
prints "30 / 30 pass · 7 not constructible on this document", never a score
that divides by cases nothing ran.

## What the renderer promises — the view rules

The refusals above prove the loader stops a wrong FILE. The **view rules**
prove the renderer keeps its own encoding law on a right one. They are
asserted on the loaded document, printed in the same self-test panel, and run
headlessly by `tests/test_qif_atlas.py`.

| # | The renderer promises |
|---|---|
| V1 | No filter repaints a group. Hue is assigned from the whole file, so what is on screen never changes what a colour means. |
| V2 | The duty channel is one neutral ink that is not a group hue. |
| V3 | Moving a macro's `duty_cycle` does not move its occupancy fill. One fact, one channel — in both directions. |
| V4 | A chip past the draw budget is not drawn, and every other chip still is. |
| V5 | Every column-map bracket covers exactly one owner tuple and its own tiles (D11). |
| V6 | A macro's breadcrumb is package → its own chip → itself. Containment, never a guess. |
| V7 | Owner search matches the owner vocabulary only. Searching a measured value selects nothing: a selection by value read as a fact about that value is the failure this rule exists to prevent. |
| V8 | One lane per system, every chip in exactly one lane, and a PD pair joined by its handoff. |
| V9 | Every metric that names a system names one this document has. |

## Query parameters

They exist so a screenshot is one URL, and so the same URL redraws the same
picture.

| Parameter | Effect |
|---|---|
| `?model=<model id>` | Which WORKLOAD the results page opens on. The default set is the bundle in `../dse/subset_v1.json`; this parameter opens any model whose artifacts exist, in or out of that bundle, and a model opened this way is appended to the selector and marked. A model with no sweep artifact raises the one band the page has. |
| `?cand=<candidate id>` | Which DESIGN POINT of that model's sweep binds the page (D33). Every panel below the frontier describes it. An id the sweep does not carry is reported in a band and the page falls back to the sweep's own initial pick. |
| `?view=package\|chip\|macro` | Open the drilldown at that level of the document bound to the selected point. Any other value is reported and ignored. |
| `?chip=<chip id>`, `?macro=<macro id>` | Which chiplet or macro to open. A macro implies its own chip. An id the bound document does not hold is reported and ignored — it is not a dead end. |
| `?duty=1` | Turn the duty-cycle overlay on. Reported and ignored on a document that declares no duty cycle. |
| `?fmodel=`, `?layer=`, `?group=`, `?pool=`, `?q=` | Pre-set the drilldown filters and the owner search. A value the bound document does not contain is reported and ignored. `?fmodel=` is the tile-owner filter; `?model=` above is the workload selector. |
| `?bundle=<name>` | Which RUN BUNDLE the page reads: `../dse/<name>.json`, default `subset_v1`. See the `fws_slot/1` section at the end of this file. |
| `?selftest=1` | Run this page's own conformance self-test — the loader refusals and the view rules — against the document bound to the selected point, and show it in the drilldown panel. It runs on request only: never on a plain load. |

The page WRITES this state back with `pushState` on every model click, point
click, descent and filter change, so the browser's Back button walks the
reading position and a copied URL redraws the same picture on the same point.

A parameter the page cannot honour — a candidate the sweep does not carry, a
chip the bound document does not hold — is dropped in silence and the page
keeps the position it had. It used to print a band about it at the top of the
page, which is a scolding for a typo.

`?data=` and `?frontier=` are GONE. They loaded a sibling document into a
one-document mode, and under D33 the drilldown is bound to the selected point:
a document chosen by URL could not state which point it described. Both are
reported in the rejected-parameter band rather than silently ignored. To read
a file that is not the selected point's, open it with the page's own file
picker — a hand-opened document is labelled as one and is never presented as
the selected point's machine.


## What the results page is, top to bottom

Nine panels, in this order, and nothing else:

1. **The design space** — the Pareto frontier (x = the headline rate, y = total
   silicon mm2) and the all-candidates table under it. This is the entry: the
   page opens here and a click here binds everything below.
2. **The two chiplet kinds of the selected point** — the shared digital card
   and the analog weight card, with the point's knobs, rate, silicon, beat and
   latency on the heading.
3. **The map of the selected point** — the pipeline stages and the drilldown
   into a chiplet and a macro.
4. **Where the beat goes** — per-stage engine, fabric and analog times.
5. **The state it holds while it runs** — the per-stage bill.
6. **Energy**, with each component's coverage.
7. **Duty cycle**, per device class.
8. **Boundaries** — what crosses between chiplets.
9. **What this point does not account for** — the disclosures, quoted verbatim.

There is no headline stat row and no provenance footer. Both printed numbers
the frontier and the cards had already printed.

**One band, one condition.** `h-alerts`, above the frontier, renders only when
the sweep artifact is missing or unparseable — without it there is no design
space, no point and no panel. Every other condition is inline where it belongs:
a legacy companion that 404s, a point document refused for naming another
candidate or another run, a query parameter the page cannot honour. The sweep
is the report and it is authoritative in silence; the page does not compare it
against `<model>_report.json` or `<model>.json` and does not comment on their
age.

## Plain language, and one fact once

- **Absence copy is one plain sentence.** The artifacts speak to the decision
  register ("EMPTY, and not a measured zero: ... so D31/ADJ-10 derived neither
  engine"). The page does not. Decision codenames are stripped from every
  string it renders except inside the disclosures panel, where the text is a
  quotation and stays verbatim.
- **An absent column is dropped, not dash-filled.** A row with no value for the
  selected point is not drawn; a table with nothing to say becomes one
  sentence; a workload with no concept of a quantity (an encoder has no beat,
  no resident streams and no KV) gets no panel and no column for it. An
  em-dash survives only where the absence is itself the finding: an uncovered
  energy term, an area no library could compose.
- **Identical rows collapse with a count.** Twenty stages holding the same bill
  are one row; 620 identical boundary crossings are one row and a count.
- **A refused point is a result.** Every panel that finds no data on an
  `ok: false` candidate prints the sweep's own refusal message instead of
  reporting a missing number.
- **An unbalanced pipeline is a finding.** When the sweep judges a point's
  stages badly matched (`stage_balance`), the beat panel leads with one plain
  paragraph naming the long stages, what makes them long, and the rate a
  balanced partition could reach. A balanced point prints nothing extra, and an
  artifact without the block prints nothing.

## The model set is a manifest

`../dse/subset_v1.json` names the models the report is about:

```
{ "bundle": "subset_v1", "generated_at": "...",
  "members": [ { "name", "model_id", "label", "mode", "dir",
                 "run_id", "selected_id", "headline" }, ... ] }
```

The selector is exactly `members`, in order, and each member's sweep is read
from `../dse/<dir>/dse_report.json` with its per-point maps under the same
directory. A checkout without the manifest falls back to the list in
`atlas.html`; a `?model=` deep link opens any model whose artifacts exist.

## The encoding law

One fact, one channel (D20, D21).

| Channel | Carries |
|---|---|
| Hue | The parallelism group, and nothing else. |
| Fill lightness | Occupancy, as a single-hue ramp inside the group's hue. |
| Bar on the macro's top edge, ink `#3d4f5c` | Duty cycle (P4), on the absolute 0–1 scale. Its own channel, off by default, never a hue and never the fill. |
| Outline style | Pool membership: solid = analog macro chiplet, dashed = shared digital chiplet. |
| Hatch | Unknown, reserved, or over-palette. Never a value. |
| Position | Containment only. Grid order comes from ids. |
| Dimming | Not selected by the filter. Never removed: position is containment, so hiding one object would move the rest. |
| Bracket over a column span | Tiles that carry ONE weight matrix: bit slices (D11), column sets, or a piece that chains to other macros. |
| Line weight | Bytes, on a √ scale. |
| Line dash | Link role. |
| `◇` | A fixture-invented value. |

### Duty cycle is drawn on the absolute scale

A duty cycle is already a fraction of the period, so the bar is `value × width`
and nothing else. Rescaling it to the busiest macro in the file would put a
second reading of one number on screen beside the first (D21), and it would
hide the actual finding: on a real run the analog macros idle for almost the
whole timeline while the shared digital engines saturate. That is D5's
prediction drawn, not a rendering fault, and the legend says so and names the
busiest macro. A macro whose `duty_cycle` is `null` draws the unknown hatch —
an absent measurement is never a zero.

### The views

| View | Draws | Descends to |
|---|---|---|
| Package | One lane per system — two of them is the PD picture (D16) — chips clustered by pool inside it, traffic-bearing links across them, the handoff labelled with its bytes | Chip |
| Chip | The chip's macros as a deterministic grid: hue = group, fill = occupancy, bottom strip = column spans, top bar = duty cycle when the overlay is on | Macro |
| Macro | The column map: every stored column of the macro, one band per resident tile at its own offset, hatch where no tile holds, brackets above (D11), and the column axis below | Tile card |
| Macro · time lanes | The same macro as a decode step: one lane per mux bank, x = ADC pass order, each pass naming the (K, N) block and the tensor that fires on it (P7.6) | Pass card |
| Folding card | One tensor, its mapping family side by side: spatial blocks as boxes, bank passes as time lanes inside a box, the accumulator drawn where the K partials meet, area/time/energy and waste under each variant (P7.6) | Pass card |
| Frontier | Total area vs decode tokens/s, one mark per complete provisioning, the declared front as a staircase, the knee ringed, and per-point utilization and waste in the table beside it (P7.6, D28) | Point card, and its atlas where one exists |

Breadcrumbs are containment and nothing else: `Package › chip › macro`. Click
descends; a breadcrumb ascends; the card of the object you descended into
stays pinned.

### Filters and owner search

`model`, `layer` (an integer or `endpoint`), `group` (an `axis:index` pair),
`pool`, and a free-text owner search over the OWNER VOCABULARY: tile, macro
and chip ids, model, block, layer, expert, slice role and group. Measured
values are deliberately not in the search index (view rule V7).

A filter changes what is drawn and never what a number says. Objects it does
not select are dimmed, never removed. The count in the filter bar is a
SELECTION count — a fact about the picture, labelled as such — and no figure
in the metric strip moves when a filter does.

### The draw budget, out loud

10 000 rects — counted as rects, which is what the drawers emit and not what a
chip declares in slots. A chip's grid emits, per macro: the macro rect, the two
duty rects when the overlay is on, and — when the macro holds tiles — the
column-strip base plus one band per resident tile. So a 640-slot chip draws
about 3 160 rects, not 640. The toolbar prints the budget and what the current
view draws, in that same unit. Above the budget only the opened chip is drawn,
and a chip whose grid alone is past the budget is not drawn either and says so
where it would have been (rule 23 warns at load, judged with the duty overlay
ON — the overlay is the reader's own toggle, so the budget has to hold in the
worst case the reader can ask for).

### The group hues

Hue is assigned to the distinct `(primary_group axis, index)` pairs present
in the file, sorted by the fixed axis order `tp, ep, pp` and then by index.
That order is a property of the file, not of what happens to be on screen, so
a filter never repaints anything. Beyond four pairs the atlas draws the
over-palette hatch and says so — it never cycles a hue.

| Slot | Hex | Use |
|---|---|---|
| 0 | `#108ca0` | first group on the primary axis |
| 1 | `#a63931` | second |
| 2 | `#714ba8` | third |
| 3 | `#808402` | fourth |

Validated with the `dataviz` skill's `validate_palette.js` against surface
`#f7f9fa`, **all-pairs** (any two macros can sit side by side in the chip
grid):

```
Lightness band       PASS   all 4 inside L 0.43–0.77
Chroma floor         PASS   all 4 >= 0.1
CVD separation       PASS   worst all-pairs ΔE 9.3 (deutan) · tritan 11.0
Normal-vision floor  PASS   worst all-pairs ΔE 17.8
Contrast vs surface  PASS   all 4 >= 3:1
```

Identity is never color alone: the legend is always on screen and every card
prints the group in words, so the figure survives grayscale print.

The board palette — ink `#1c2a33`, teal `#176778`, paper `#f7f9fa` — stays
chrome: headings, rules, panels. `#176778` sits below the chroma floor
(0.077) and cannot do identity work, so it is never a group hue.

## The documents this contract's own tests pin

Other producers write their documents into this directory too; each has to
pass the loader, which is the conformance check the contract gives for free.
The four below are the ones `tests/test_qif_atlas.py` holds to the schema.

| File | What it is |
|---|---|
| `fixture_llama7b_tp2.json` | The P5.1 phase-0 fixture: a hand-authored Llama2-7B tp = 2 decode placement with its invented fields marked. It is what the loader self-test is proved against. The page does not open on it and does not embed it. |
| `p3_llama7b_tp2.json` | The same model emitted by the real producer, placement only (P3.6). |
| `granite_4_0_h_tiny.json` | The ADJ-1 headline artifact: Granite-4.0-H-Tiny, placed and priced (`--priced`). |
| `fixture_granite_folding.json` | The P7.6 folding fixture: a real Granite excerpt carrying the `folding` and `passes` blocks. |
| `fixture_frontier_granite.json` | The P7.6 frontier fixture (`fws_frontier/1`, not an atlas document). |
| `pd_llama7b.json` | The PD pair (D16): a prefill inventory and a decode inventory with different layer packings and chiplet counts, one priced handoff, both halves priced on their OWN timeline. It is the two-system picture. |

Each exported document names in `provenance.reference_run.command` the exact
command that regenerates it, and each is pinned by a regenerate-and-compare
test — a stale artifact is a wrong picture under a right title.

## The page embeds nothing

`atlas.html` used to carry a copy of the fixture in a
`<script type="application/json" id="atlas-embedded">` block, with a second
one for a frontier document, and a `window.RS_PRELOADED` branch that
short-circuited every fetch. All three are GONE. An embedded document is a
stale document the moment the artifacts move, and the two blobs had been null
for long enough that the "re-embed it when the fixture changes" instruction
described nothing. The page fetches its artifacts on every load and names
each one, its HTTP status and its run id in the provenance footer; a fetch
that fails is a visible band naming the URL and the status, not an empty
panel.

The page needs http to read those artifacts. Opened from a `file://` URL it
cannot fetch at all, and it says so in the failure band instead of drawing an
empty design space.

# The P7.6 folding extensions

> **`atlas.html` no longer implements any of this.** D33 makes the results
> page a frontier and a selected point, and the three views below never drew
> on a shipped artifact: no document this repository produces has ever carried
> a `folding` or a `passes` block, and no `fws_frontier/1` document is fetched
> anywhere. The viewer's loaders, corruption harnesses and drawings for them
> (about 1,300 lines) were cut with the rest of the dead layer. The three
> contracts below stand as DATA contracts for their Python producers and their
> fixtures; the sentences in this part that say "the loader refuses", "the
> page draws" or "the self-test panel prints" describe a viewer that no longer
> exists. `tests/test_qif_atlas.py` still exercises them against
> `ATLAS.validateFold` / `validatePasses` / `validateFrontier`, which are gone
> from the page — see the blocker note in the rebuild report.

Three views were added for P7 (folding): the **folding card**, the **macro
time-lane view**, and the **frontier explorer**. They read three contracts.
Two of them are OPTIONAL blocks inside an `fws_atlas/1` document — a document
without them loads and draws exactly as before — and the third is a document of
its own, because a frontier is many provisionings and an atlas is one.

| Contract | Where it lives | What it draws |
|---|---|---|
| `fws_fold/1` | `folding` — an optional top-level block of an `fws_atlas/1` document | The folding card: one tensor, its mapping family side by side |
| `fws_pass/1` | `passes` — an optional top-level block of an `fws_atlas/1` document | The macro drill-in: a macro's banks as time lanes of one decode step |
| `fws_frontier/1` | Its own document | The frontier explorer: total area vs decode tokens/s |

Every one of them is **decode only (D25)** and says so in its own text, and the
loader refuses an artifact that does not. The two rules that govern the atlas
govern these too: the loader **computes only to refuse**, and a fold that
cannot be drawn is not a fold. An error in an extension block stops the whole
page drawing, exactly as an error in the placement tables does — the picture
is one artifact and it is right or it is refused.

## The `folding` block (`fws_fold/1`)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema` | string | yes | Must be `"fws_fold/1"`. |
| `decode_only` | string | yes | The decode-only assumption, in words (D25). |
| `invariant_w` | string | no | The Invariant W statement the card is drawn under (D27). |
| `refused[]` | array | yes | The DOA register, carried IN the artifact: `{fold, reason, decision}`. Every name in the register below must appear. |
| `cards[]` | array | yes | One folding card each. |
| `fixture_invented` | bool \| string[] | no | The usual mark. |

### The DOA register (D26)

`refused[]` must name all seven by name. D26 makes the register binding, so an
artifact that quietly drops an entry is the one place a dead fold could come
back.

| `fold` | Why it is dead |
|---|---|
| `slicing_x_folding` | No shipped card slices (`bits_per_cell` is null), so the product has nothing to be about. |
| `replication` | A second copy idles or duplicates weight space decode cannot use (D27). |
| `non_canonical_walk` | Splitting K when K ≤ the card's rows buys nothing: one column already holds the whole K. |
| `prefill_folding` | Prefill is out of P7 (D25). |
| `multi_tenant_fold` | The same machinery later, named and empty today (D8). |
| `pipelined_decode_fold` | Overlapping decode steps leaves D15's regime. |
| `idle_spread` | Spreading past the cell floor idles banks (D27). |

A variant whose `kind` is any of these is refused BY NAME. The kinds that are
alive in decode are `n_spread`, `k_spread`, `k_stack`, `dense_share` and
`row_band`.

### `folding.cards[]`

| Field | Type | Meaning |
|---|---|---|
| `id`, `label` | string | Identity. |
| `tensor` | object | `{owner, label, K, N, cells, basis}`. `owner` is the atlas owner tuple `{model, layer, block, expert}` and must exist in the document's owner vocabulary. `cells` must equal `K × N`. |
| `card` | string | The device card the fold is drawn on; must exist in `cards`. |
| `floor_macros` | int | `ceil(cells / (rows × stored_cols))` — the tensor's share of the GLOBAL cell floor (D27). Checked. |
| `floor_basis` | string | The arithmetic behind it. |
| `variants[]` | array | At least two: a card draws the mapping FAMILY, not one mapping. |

### `folding.cards[].variants[]`

| Field | Type | Meaning |
|---|---|---|
| `id`, `label`, `note` | string | Identity and the sentence under the drawing. |
| `kind` | string | One of the five live folds; a DOA name is refused. |
| `boxes[]` | array | The spatial blocks. See `box` below. |
| `accumulator` | object | `{where, operands, sites[], transport_bytes, per, basis}`. `where` is `none` \| `in_macro` \| `cross_macro`. A variant that places more than one K block of its tensor and declares `none` is refused: K partials have to meet somewhere. `sites[]` name boxes OF THIS VARIANT. `transport_bytes` must be 0 when `where` is `in_macro`. |
| `waste` | object | `{cells, denominator_cells, pct, terms, basis}`. `denominator_cells` is the committed cells the percentage is taken over and `pct` must equal `cells / denominator_cells × 100`. `terms` is optional and, when present, is P7.2's own itemization — `{remainder_cells, tail_cells}`, which must sum to `cells` and have no third term, exactly as `cim_timing.dense_pack` reports it. D27 makes waste a REPORTED metric and this is where it is reported. |
| `area` | object | `{basis_kind, value, unit, basis, terms, uncovered[]}`. `basis_kind` is `slots` (with an integer `slots`) or `cell_share` (with an integer `cells`). `value` is checked against that accounting: `slots × card.area_mm2`, or `cells / (rows × stored_cols) × card.area_mm2`. When the card declares no `area_mm2` the value must be `null` and draws as the unknown hatch. |
| `time` | object | `{passes, value, unit, per, basis}`. `passes` counts THIS tensor's ADC passes — a co-tenant sharing a bank pays its own pass — and is checked against the enumerated lanes. `value` may be `null`. |
| `energy` | object | `{value, unit, uncovered}`. `value` may be `null`, and then `uncovered` must name the ABSENT law. A silent zero is refused: an uncovered term and a measured zero are not the same statement. |

### The `box` (shared by both blocks)

A box is one macro — real or, in a fold that no mapper has placed, hypothetical
— drawn as its card's mux banks.

| Field | Type | Meaning |
|---|---|---|
| `id`, `label` | string | Identity. |
| `macro` | string \| null | The macro this box IS. `null` means the box is a fold, not a placement, and it draws with a dashed outline. |
| `card` | string | Must exist in `cards`; in a `passes` box it must equal the macro's own card. |
| `passes` | int | The enumerated pass count of this box. Checked. |
| `lanes[]` | array | **Every** bank of the card, whether it fires or not — so weight space a schedule leaves uncovered is visible rather than omitted (D27). |

`lanes[]` entries are `{bank, columns, passes[]}`. `bank` is an index in
`[0, mux)`, drawn once; `columns` must be exactly that bank's span,
`{start: bank × cols_adc, count: cols_adc}`.

`lanes[].passes[]` entries:

| Field | Type | Meaning |
|---|---|---|
| `pass` | int | The pass index inside ONE decode step. Unique within its lane: one bank does one thing at one moment. |
| `label` | string | What fires. |
| `owner` | object | The tensor's owner tuple. |
| `tile` | string \| null | The placed tile. Required in a `passes` box, where it must be a tile the macro holds and must contain the pass's columns. `null` in a fold box. |
| `k_block`, `n_block` | int | Which block of the tensor fires. The pair identifies a block within a variant, so two passes claiming the same pair are the same weights stored twice — which is replication, and refused (F7). |
| `rows`, `columns` | object | Real spans: `rows` inside the card's rows, `columns` inside the lane's own bank. |
| `co_tenant` | bool | True when the owner is not the card's tensor. |

## The `passes` block (`fws_pass/1`)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema` | string | yes | Must be `"fws_pass/1"`. |
| `phase` | string | yes | Must be `"decode"`. A prefill schedule is refused by name (D25). |
| `label` | string | no | One line. |
| `step` | object | yes | `{index, of, basis}` — which decode step, out of what lowered window. |
| `pass_time_s` | number | no | Seconds per ADC pass. |
| `pass_time_basis` | string | with the above | Where that number came from. |
| `boxes[]` | array | yes | One box per macro, in the shape above, each naming a macro this document places. |

## The frontier document (`fws_frontier/1`)

The contract P7.4/P7.5 emit and the frontier explorer reads. It is a separate
file: `?frontier=<basename>` loads one, and dropping one on the page adds it
without displacing the atlas.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema` | string | yes | Must be `"fws_frontier/1"`. |
| `title`, `subtitle` | string | title yes | One line each. |
| `provenance` | object | yes | Same shape as the atlas's: `authored`, `author`, `reference_run`, `grounded[]`, `invented_fields[]`, `note`. |
| `decode_only` | string | yes | The decode-only assumption, in words (D25). |
| `provisioning_stance` | string | no | The no-margins statement (D28). |
| `model` | object | yes | `{id, label, basis}`. |
| `axes` | object | yes | `x` and `y`, each `{key, label, unit, basis}`. |
| `device_classes[]` | array | yes | `{id, label, basis}`. Every point reports utilization for every one of them (D28). |
| `knobs[]` | array | no | `{id, label, basis}` — the sizing axes swept. |
| `points[]` | array | yes | The provisionings. |
| `knee` | object | yes | `{point, basis, caveat}` — which point is the knee and the law that picked it. |
| `relaxations[]` | array | yes | `{constraint, value, reason}`, as in the atlas. |

### `points[]`

| Field | Type | Meaning |
|---|---|---|
| `id`, `label` | string | Identity. |
| `in_sweep` | bool | Whether the point came from the swept cross product. |
| `knobs` | object | The sizing values this point is. |
| `provisioning` | object | `{analog_chips, shared_chiplets, analog_macro_slots, placed_tiles, basis}`. Integers: a per-chip quotient here is refused (D21). |
| `area` | object | `{total_mm2, terms, uncovered[], basis}`. The finite terms must sum to `total_mm2`; a term with no law is `null` and contributes nothing, and says so. |
| `throughput` | object | `{tokens_per_s, decode_step_s, basis}` — read off this point's OWN timeline (ADJ-6). |
| `utilization[]` | array | One entry per declared device class: `{device_class, value, uncovered, basis}`. `value` is a fraction in `[0, 1]`, or `null` with `uncovered` naming the absent measurement. D28 makes this mandatory at every point, so idle silicon is always visible. |
| `packing` | object | `{waste_pct, basis_kind, basis}` — the weight space this packing wastes (D27). Mandatory. |
| `atlas` | object \| null | `{document, exists, basis}`. `document` is a bare basename in the atlas's own directory; a point with `exists: false` names none. Only a point whose artifact exists is clickable. |
| `pareto`, `knee` | bool | Declared flags. The loader checks them against the enumerated points and never invents them. |

**No safety margins, by name (D28).** Any field anywhere in the document whose
name matches `margin`, `derate`, `guard_band` or `safety` is refused. The
frontier itself is the statement of how close a point runs; a margin would hide
it.

## What the extension loaders refuse to draw

Same discipline, separate rule series so the two contracts stay legible. F and
T rules are emitted by `ATLAS.validateExt`; X rules by
`ATLAS.validateFrontier`.

| # | Check | Refuses when |
|---|---|---|
| F1 | `folding.schema` / shape | The schema id is not `fws_fold/1`, or a card draws fewer than two variants, or a variant draws no boxes. |
| F2 | `folding.decode_only` | The artifact does not state the decode-only assumption (D25). |
| F3 | `folding.refused[]` | The DOA register is missing, an entry lacks `fold`/`reason`/`decision`, or one of the seven dead folds is not refused by name (D26). |
| F4 | `variants[].kind` | The kind is a DOA fold — refused BY NAME — or is not one of the five live folds. |
| F5 | `cards[].tensor` | The owner names a model or block this document does not have, `K`/`N` are not positive integers, `cells ≠ K × N`, the tensor states no basis, or the device card does not exist. |
| F6 | `cards[].floor_macros` | It disagrees with `ceil(cells / (rows × stored_cols))`. Analog silicon is the model's floor, not the fold's (D27). |
| F7 | `variants[].boxes` | The enumerated blocks do not cover the tensor's cells exactly once — including the case where one `(k, n)` block is placed twice, which IS replication. |
| F8 | `variants[].area` | `basis_kind` is outside `slots`/`cell_share`, the value disagrees with that accounting, or a number is declared where the card declares no `area_mm2`. |
| F9 | `variants[].time.passes` / `boxes[].passes` | A declared pass count is not what the variant or the box draws. |
| F10 | `variants[].energy` | A null energy names no absent law, or a value is not finite. |
| F11 | `variants[].waste` | The waste block is missing or malformed, `pct` is not `cells / denominator_cells × 100`, or an itemization is carried whose `remainder_cells + tail_cells` is not the waste (D27, P7.2). |
| F12 | `variants[].boxes[].lanes` | The lanes are not the card's banks, a bank is drawn twice, a lane's columns are not its bank's span, two passes share a pass index, or a pass's rows/columns fall outside its card or its own bank or its own tile. |
| F13 | `variants[].accumulator` | `where` is outside the vocabulary, a multi-K variant declares no accumulator or fewer than two operands, a site names a box the variant does not draw, or an `in_macro` accumulator claims transport bytes. |
| T1 | `passes.schema` / shape | The schema id is not `fws_pass/1`, or the block draws no boxes. |
| T2 | `passes.phase` | The phase is not `decode`. A prefill schedule is refused by name (D25). |
| T3 | `passes.step` | The schedule does not say which decode step it is and out of what window, or a per-pass time arrives with no basis. |
| T4 | `passes.boxes[].macro` | The macro does not exist, or the box's card is not the macro's card. |
| T5 | `passes.boxes[].lanes` | Same lane law as F12: every bank is drawn, once, as its own span. |
| T6 | `passes.boxes[].lanes[].passes[].tile` | The pass names no tile the macro holds, or drives columns its own tile does not own. |
| T7 | `passes.boxes[].lanes[].passes[].pass` | Two blocks fire on one bank at one pass. |
| T8 | `passes.boxes[].passes` | The declared pass count is not what the box draws. |
| T9 | *(warning)* | A bank of a macro fires on no pass of this step. Under Invariant W every bank holds real weights and fires once per step, so this is WASTE — reported, never refused (D27). |
| X1 | `schema` / `title` / `provenance` | The schema id is not `fws_frontier/1`, or the document has no title, no provenance, or a provenance without `authored`, `author` and `invented_fields[]`. |
| X2 | `decode_only` | The frontier does not state the decode-only assumption (D25). |
| X3 | `axes.x` / `axes.y` | An axis does not name its key, its label and its unit. |
| X4 | `device_classes[]` | The document reports no device classes, or one has no id/label (D28). |
| X5 | `points[].area` | A point states no total or no basis, a term is neither finite nor null, or the terms do not sum to the total. |
| X6 | `points[].throughput` | The throughput is missing, not positive, or states no basis. |
| X7 | `points[].utilization` | A point reports no utilization for a declared class, names a class that is not declared, gives a value outside `[0, 1]`, or leaves an absent one silent instead of naming the missing measurement (D28). |
| X8 | `points[].packing` | A point does not report the weight space its packing wastes, as a percentage with a basis (D27). |
| X9 | `points[].pareto` | A flagged point is dominated by another, or an unflagged point is dominated by none. |
| X10 | `points[].knee` / `knee` | There is not exactly one knee, the knee is not on the front, or the document's `knee.point` names a different one. |
| X11 | *(any field)* | A safety margin appears under any of its names (D28). |
| X12 | `points[].atlas` | The link is malformed, names a path instead of a sibling basename, or claims a document while declaring `exists: false`. |
| X13 | `points[].provisioning` | A provisioning arrives as a per-chip quotient (D21). |
| X14 | `relaxations[]` | An entry is missing `constraint`, `value` or `reason`. |

`atlas.html` carries a self-test for each series, on the same terms as the
atlas's own: corrupt a copy of the loaded document once per rule and assert the
refusal fires and names the right field. The folding self-test runs 24 cases
(the document as loaded plus one per corruption) and the frontier self-test
runs 17. Both are reachable from the same **Self-test** button and the same
`?selftest=1`, and `tests/test_qif_atlas.py` runs both headlessly, so a rule
that stops firing fails the suite rather than a screenshot. A document that
carries no `folding` or `passes` block reports all 23 folding cases as `n/a`,
"not constructible on this document" — which is the honest reading, and it is
why `fixture_granite_folding.json` exists.

## The extension view rules

| # | The renderer promises |
|---|---|
| V10 | A folding variant carries no colour of its own. Hue stays the tensor's parallelism group, and the mapping family is told apart by position and label. |
| V11 | A time lane is one bank and its x axis is pass order inside ONE decode step: nothing fires twice at one moment and no pass spills out of its own bank. |
| V12 | Every drawn frontier point carries its own per-device utilization and its own waste, and the front and the knee are flags the explorer DRAWS, never values it computes. |
| V13 | A frontier point is clickable only when its artifact exists; a point with none is drawn as unavailable and says why. |

V10 to V13 are no longer emitted by anything: `ATLAS.viewTest` returns V1 to
V9 only, and `frontierViewTest` is gone with the rest of the extension layer.

V10 and V11 were emitted by `ATLAS.viewTest` only for a document that carries
the blocks — a rule with nothing to judge would be a pass that proved nothing.
V12 and V13 are emitted by `ATLAS.frontierViewTest` on a frontier document.

## What the results page is, top to bottom

Nine panels, in this order, and nothing else:

1. **The design space** — the Pareto frontier (x = the headline rate, y = total
   silicon mm2) and the all-candidates table under it. This is the entry: the
   page opens here and a click here binds everything below.
2. **The two chiplet kinds of the selected point** — the shared digital card
   and the analog weight card, with the point's knobs, rate, silicon, beat and
   latency on the heading.
3. **The map of the selected point** — the pipeline stages and the drilldown
   into a chiplet and a macro.
4. **Where the beat goes** — per-stage engine, fabric and analog times.
5. **The state it holds while it runs** — the per-stage bill.
6. **Energy**, with each component's coverage.
7. **Duty cycle**, per device class.
8. **Boundaries** — what crosses between chiplets.
9. **What this point does not account for** — the disclosures, quoted verbatim.

There is no headline stat row and no provenance footer. Both printed numbers
the frontier and the cards had already printed.

**One band, one condition.** `h-alerts`, above the frontier, renders only when
the sweep artifact is missing or unparseable — without it there is no design
space, no point and no panel. Every other condition is inline where it belongs:
a legacy companion that 404s, a point document refused for naming another
candidate or another run, a query parameter the page cannot honour. The sweep
is the report and it is authoritative in silence; the page does not compare it
against `<model>_report.json` or `<model>.json` and does not comment on their
age.

## Plain language, and one fact once

- **Absence copy is one plain sentence.** The artifacts speak to the decision
  register ("EMPTY, and not a measured zero: ... so D31/ADJ-10 derived neither
  engine"). The page does not. Decision codenames are stripped from every
  string it renders except inside the disclosures panel, where the text is a
  quotation and stays verbatim.
- **An absent column is dropped, not dash-filled.** A row with no value for the
  selected point is not drawn; a table with nothing to say becomes one
  sentence; a workload with no concept of a quantity (an encoder has no beat,
  no resident streams and no KV) gets no panel and no column for it. An
  em-dash survives only where the absence is itself the finding: an uncovered
  energy term, an area no library could compose.
- **Identical rows collapse with a count.** Twenty stages holding the same bill
  are one row; 620 identical boundary crossings are one row and a count.
- **A refused point is a result.** Every panel that finds no data on an
  `ok: false` candidate prints the sweep's own refusal message instead of
  reporting a missing number.
- **An unbalanced pipeline is a finding.** When the sweep judges a point's
  stages badly matched (`stage_balance`), the beat panel leads with one plain
  paragraph naming the long stages, what makes them long, and the rate a
  balanced partition could reach. A balanced point prints nothing extra, and an
  artifact without the block prints nothing.

## The model set is a manifest

`../dse/subset_v1.json` names the models the report is about:

```
{ "bundle": "subset_v1", "generated_at": "...",
  "members": [ { "name", "model_id", "label", "mode", "dir",
                 "run_id", "selected_id", "headline" }, ... ] }
```

The selector is exactly `members`, in order, and each member's sweep is read
from `../dse/<dir>/dse_report.json` with its per-point maps under the same
directory. A checkout without the manifest falls back to the list in
`atlas.html`; a `?model=` deep link opens any model whose artifacts exist.

## The encoding law in the folding views

The law does not change. Hue is still the parallelism group and nothing else;
these views add channels rather than borrowing that one, and the frontier uses
one channel FEWER.

| Channel | Carries |
|---|---|
| Lane (a row inside a box) | One mux bank of the card. Every bank gets a lane, whether it fires or not. |
| x inside a box | ADC pass order within ONE decode step (D25). It is not a column offset; the column span is printed on the card beside it. |
| Outline of a pass block | Co-tenancy: solid = the card's own tensor, dotted = another tensor sharing the bank. |
| Hatch inside a pass block or a lane | The part of a bank no pass drives — waste, reported (D27). |
| Chevron glyph, ink `#3d4f5c` | The accumulator, drawn where the K partials meet. In-macro inside the box, cross-macro between boxes, with the transported bytes named. |
| Dashed box outline | A box with no `macro`: a fold the mapper has not placed. |
| Frontier: ink lightness | On the declared front (`#1c2a33`) or dominated (`#8497a2`). Both are the same neutral: no hue appears in the frontier view, because a design point has no parallelism group. |
| Frontier: mark shape | A filled square is a point with a placement document to open; a circle is a point without one. |
| Frontier: ring | The knee. |

## The documents the folding views are proved against

| File | What it is |
|---|---|
| `fixture_granite_folding.json` | The P7.6 hand fixture: eight real macro slots of `granite_4_0_h_tiny.json` — chips, macros, tiles, cards and the link byte law copied row for row — plus hand-authored `folding` and `passes` blocks built from those very shapes. It is an EXCERPT and says so in `relaxations[]`; its `excerpt.` metrics are facts about the excerpt and its `run.` metrics name the whole-model run they were quoted from. `tests/test_qif_atlas.py` asserts the copied rows are byte-identical to the source artifact, which is the anti-staleness gate a hand fixture can have. |
| `fixture_frontier_granite.json` | The P7.6 frontier fixture: the eight REAL candidate rows of `docs/qif/dse/granite_lanes_banks/dse_report.json` plus the shipped Granite atlas run as a ninth point, the only one with an artifact to click through to. Its silicon is fully composed and carries all THREE terms — analog macro slots, shared digital chiplets and per-macro digital pools, the last added by P7.9 — so `area.total_mm2` equals the source sweep's own `total_silicon_mm2` point for point and `invented_fields` is empty. ONE field is still marked: the per-candidate shared-digital UTILIZATION, which no producer emits. Two relaxations name what the document is rather than what it lacks: its waste figure is D27's CELL census of ONE reference mapping repeated on every row (the per-point census is emitted by the D31-legal curves under `docs/qif/dse/*_frontier/`), and its eight swept points sweep `vector_lanes`, the axis D31 retires — they are kept as the frozen Wave-D demo, while the ninth point's widths are ALL DERIVED — ADJ-10 derives its attention fabric (`num_arrays`, `softmax_lanes`) as well as its scan engine, so it carries a `num_arrays_provenance` of `derived-count` that no swept point has, and it is the fastest point on the picture rather than the dominated one ADJ-9 left. It is the shape `fws_frontier/1` producers emit. |

# The `fws_slot/1` slot-map extension

D39 (2026-08-31). A macro of X rows x Y columns behind Z ADCs rotates through
`mux = Y/Z` slots once per beat, so **a slot index is a firing phase**: the
slot-layer sitting in slot j fires on every tick that is j modulo mux. Where a
weight sits and when it fires are ONE decision, and `slots` is that decision
written down.

The block rides on a normal `fws_atlas/1` document. Everything the base
contract requires is still required and still refused the same way; `slots` adds
the schedule the base contract has no words for. A document without it is
unchanged, and a reader without the slot view still sees a conformant map.

    "slots": {
      "schema": "fws_slot/1",
      "platform": {...}, "chips": [...], "macros": [...],
      "bundles": [...], "ledgers": {...}, "check": {...}
    }

Ids are strings, zero-padded, and sorted id order IS layout order:
`chip.00`, `mac.000`, `b.000`. Ticks are integers and they are TOKEN 0's
firing ticks, counted from tick 0 of the first bundle.

## `slots.platform`

One object. It states the machine the schedule runs on, and every number in it
is fixed by the platform point, not by the mapper.

| Field | What it is |
|---|---|
| `rows` | X: the macro's row count. In the simple universe it equals the model's hidden size. |
| `z` | Z: how many ADCs a macro has. One slot-layer is Z columns wide. |
| `y` | Y: the macro's stored columns. `y = mux * z` or the loader refuses. |
| `mux` | Y/Z, DERIVED. The number of slots a macro rotates through, and the number of firings it owes per token. |
| `tick_ns` | One ADC firing. 20 ns. |
| `beat_ticks`, `beat_ns` | The rotation: `beat_ticks = mux`, `beat_ns = beat_ticks * tick_ns`. Throughput is one token per beat and nothing the mapper does moves it. |
| `hop_gap_ticks` | Ticks between consecutive bundles: one to finish converting, one for the wire and the reduce. 2. |
| `seam_ticks`, `seam_ns` | What a chip crossing adds on top of the hop. 25 ticks, 500 ns. |
| `chip_macros` | How many macros a chip holds. |
| `macro_area_mm2` | The area of ONE macro of this platform, from the sheet: `cim_analog_calc.derive_analog_card(rows, y, z)`. |
| `label` | The platform's name, `512x2048:64`. Shown as written. |

## `slots.chips[]` and `slots.macros[]`

Containment is stated in both directions, exactly as the base contract states
it for `chips[]`/`macros[]`, and the loader checks both.

| Field | What it is |
|---|---|
| `chips[].id`, `.index` | The chip and its position left to right. |
| `chips[].macros[]` | Its macro ids, in column order. |
| `chips[].capacity_macros` | How many macros the chip holds. It equals `macros.length`. |
| `macros[].id`, `.chip`, `.index` | The macro, the chip that holds it, and its column position WITHIN that chip. |
| `macros[].slots[]` | One entry per OCCUPIED slot (see below). |
| `macros[].free_slots[]` | Slot indices the rotation reaches and nothing fires: capacity this map did not need. |
| `macros[].stranded_slots[]` | Slot indices left behind when the next bundle did not fit this chip and moved whole. Their cells are WASTE. |

Occupied, free and stranded together account for exactly `mux` slots, each
index once.

### `macros[].slots[]` — one slot-layer

| Field | What it is |
|---|---|
| `slot` | The slot index, 0..mux-1. It is the firing phase. |
| `bundle` | The bundle id this slot-layer belongs to. |
| `layer`, `block` | The model layer number and the weight block, for colour and hatch. |
| `kblock` | Which K-block of the bundle, 0 when the bundle has one. |
| `columns` | `{start, count}` — the span this slot-layer holds WITHIN its bundle's column range. `count < z` is a ragged slot-layer, and `z - count` columns of the ADC set hold nothing. |
| `tick` | Token 0's firing tick. `tick % mux == slot` or the loader refuses. |

## `slots.bundles[]`

A bundle is one weight matrix of one layer: the unit that moves whole, is
never split across chips, and hops to the next bundle in chain order.

| Field | What it is |
|---|---|
| `id`, `order` | The bundle and its position in the chain. Sorted order is chain order. |
| `layer`, `block` | Which layer and which weight block. |
| `columns` | Its real columns, ALL K-BLOCKS TOGETHER. The K-blocks share the range equally and own contiguous chunks of it, so `columns` divides by `k_blocks` and K-block k holds `[k * columns / k_blocks, (k+1) * columns / k_blocks)`. This is the figure the chain prints beside the slot-layer count, and the two only agree on this reading. |
| `k_blocks` | How many row-blocks the matrix is cut into. Above one their partials REDUCE, and they may fire at the same tick on different macros because each has its own input slice. |
| `reduce` | True when those partials are summed. |
| `slot_layers` | How many slots it occupies. It equals the map's own count. |
| `ragged_columns` | Columns of committed ADC width that hold no weight. |
| `chip`, `macros[]` | Where it sits. One chip, always. |
| `start_tick`, `end_tick` | Token 0's first and last firing of this bundle. |
| `serial_ticks` | `end_tick - start_tick + 1`. |
| `stall_ticks` | How long the bundle waited past the earliest tick the law allows: `start_tick` minus the previous bundle's `end_tick` plus the hop gap, plus the seam when `seam_before`. It is a wait BEFORE the span, not idle ticks inside it. Its floor is 0. |
| `seam_before` | True when it sits on another chip than the bundle before it, which is what buys the 25-tick seam. |

## `slots.ledgers`

The three ledgers the mapper controls are cell waste, chain latency (and the
sequences it puts in flight) and seams. THE ATLAS READS THESE VERBATIM. It
recounts them only to disagree out loud (rule S9); it never substitutes its
own figure for the producer's.

| Field | How it is counted |
|---|---|
| `beat_ticks`, `beat_ns` | The platform's, repeated here so a ledger reads on its own. |
| `L_ticks`, `L_ns` | The chain latency: the last firing tick of the last bundle, plus the one tick that converts it, counted from tick 0. |
| `L_min_ticks` | The tight floor the law permits: `2n - 1` ticks for `n` bundles (the 2-tick hop gap separates the `n - 1` joins, and `L` counts the last firing plus one), plus 25 for each seam that chip capacity forces. `L_ticks` is what the map achieved; the gap between the two is stalls and temporal depth. |
| `D`, `D_min` | Little's law: `ceil(L / beat_ticks)` sequences in flight, and the same over the floor. |
| `macros_used`, `chips_used` | Counted from the enumerated map. |
| `slots_total` | `macros * mux` over every macro in the map. |
| `slots_used`, `slots_stranded` | Occupied and stranded slots. Free slots are the remainder. |
| `cells_total` | `slots_total * z * rows`: every cell the map commits. |
| `cells_used` | Cells holding real weights: the slot-layers' own columns times rows. |
| `cells_wasted` | Ragged cells plus every stranded slot's cells. Free slots are NOT waste: nothing was displaced to leave them. |
| `waste_frac` | `cells_wasted / cells_total` — a share of EVERY cell the mapping commits, free slots included, which is the denominator the atlas prints beside it. Its floor is 0. |
| `seams` | Bundles with `seam_before`. |
| `stall_ticks` | The bundles' stalls, summed. |
| `throughput_tokens_per_s` | `1 / beat`. Occupancy only. |
| `per_user_tokens_per_s` | `1 / (D * beat_ns)`, which is `throughput / D`. A user's next token re-enters on the first bundle's own phase, and that phase comes round once per beat, so one user's period is D beats. Little's law closes on the strip: `D` times this rate is the throughput, and no user out-runs the machine. |
| `per_user_ceiling_tokens_per_s` | `1 / (L_ticks * tick_ns)`: the rate a user would get if the next token could start the instant the last one left. It is a CEILING, not a rate. Only a beat that divides `L_ticks` reaches it. |
| `analog_area_mm2` | `macros_used * macro_area_mm2`. The toy has no other silicon term. |

## `slots.check`

`{"ok": bool, "violations": []}` — the verdict of the INDEPENDENT checker that
reads the emitted document alone. `ok: false` is not a warning: the atlas draws
no map at all and prints the violations. Every emitted mapping carries it.

## What the slot loader refuses to draw

S1-S8 are refusals: the section draws nothing and prints the rule, the field
and the sentence. S9 is a disagreement, printed under the ledgers with the
producer's own numbers left standing. They live in `ATLAS.validateSlots(doc)`
inside the headless core block, beside the base loader.

| Rule | What it refuses |
|---|---|
| S1 | A `slots` block that is not `fws_slot/1`, or that is missing `platform`, `chips[]`, `macros[]`, `bundles[]`, `ledgers` or a `check` of `{ok, violations[]}`. |
| S2 | A platform that contradicts itself: a non-positive geometry, `y != mux * z`, `beat_ticks != mux`, or a `beat_ns`/`seam_ns` that is not its tick count times `tick_ns`. |
| S3 | Containment stated one way only: a chip listing a macro that does not name it back, a macro naming a chip that does not hold it, or a `capacity_macros` that is not the number of macros listed. |
| S4 | A rotation that does not add up: a slot index outside 0..mux-1, a slot filled twice, a slot-layer holding 0 or more than Z columns, or occupied + free + stranded that is not exactly `mux` slots each named once. |
| S5 | A firing tick that is not the slot's own phase (`tick % mux != slot`). Placement and schedule are one decision, and a document that separates them describes a machine that cannot run. |
| S6 | One macro firing twice at the same tick. One ADC set, one firing, one tick. |
| S7 | A bundle the map contradicts: no slots at all, a `slot_layers` count the map does not hold, a span the map does not fire, a `chip` the map does not put it on, or a bundle split across two chips. It also refuses a column count the map does not lay down: `columns` must divide by `k_blocks`, `slot_layers` must be `ceil(columns / k_blocks / z) * k_blocks`, and the slot-layers of K-block k must cover that K-block's own contiguous chunk of `0..columns` exactly once. Without it a document can print one width in the chain and store another in the grid, and each looks right on its own. |
| S8 | A gap the law forbids: a first bundle that does not start at tick 0, a next firing sooner than the previous bundle's last firing plus the hop (plus the seam when the chip changes), or a `seam_before` that disagrees with whether the chip actually changed. |
| S9 | (warning) A ledger the enumerated slots count differently, `waste_frac` included: the percentage and the sentence under it are the same division, so a reader who does the arithmetic cannot catch the page out. The figure on screen stays the producer's; the difference is printed under it. |

## What the slot view draws

| Encoding | Meaning |
|---|---|
| Cell fill | The LAYER number, on the same monotone hue ramp the rest of the page uses. |
| Cell hatch | The weight block, from the same block-order hatch table. |
| Cell width | A full cell is Z columns. A narrower fill is a ragged slot-layer, and the remainder is drawn in the ragged texture. |
| Very light cell | A free slot. |
| Textured neutral cell | A stranded slot. |
| Grid row | The slot index, slot 0 on top. It is also the firing phase. |
| Grid column | One macro. Columns are grouped into the labelled box of the chip that holds them. |
| Timeline mark | One firing of TOKEN 0 at (tick, macro), in the layer's hue. |
| Faint timeline mark | The same firing for TOKEN 1, one beat later. The overlap is the throughput. |
| Dashed vertical rule | A beat boundary, labelled beat 1, beat 2, ... |
| Shaded span | The gap between two bundles: the hop, or the 25-tick chip crossing, labelled with its 500 ns. |
| Bracket above the marks | One bundle's span, in chain order. |
| Dashed accent rule | L, with the sequences in flight beside it. |

The grid and the timeline scroll INSIDE their own container; the page never
scrolls sideways. A mux of 128 stays legible: the cells shrink and the slot
axis labels thin out, and no cell is ever dropped.

Clicking a bundle — in the chain strip, the slot map or the timeline — follows
it through all three; clicking it again clears. The selection is a view state
and is not written to the URL.

## Query parameter

| Parameter | Effect |
|---|---|
| `?bundle=<name>` | Which RUN BUNDLE the page reads: `../dse/<name>.json`. The default is `subset_v1`, the model report. A bundle named here and missing from the checkout raises the one band the page has, naming the URL — the page does not fall back to another bundle's models, because that would answer a question the reader did not ask. The name is a file name in `../dse`, never a path. A small link beside the model selector switches between the model report and the toy slot map (`toy_slotmap`) and back, and `stateURL()` carries the bundle so a copied link opens the same report. |

A member of a bundle whose `mode` is `TOY` is a mapper debug run. It renders on
the same page in the same order — frontier first, then the hardware cards, then
the map — and the panels it has no data for state their absence in the words
they already use. The toy has NO digital chiplet and NO per-sequence state
store, so the digital card says so in one sentence and no zero is printed
anywhere for either.

## The documents the slot view is proved against

| File | What it is |
|---|---|
| `fixture_slotmap.json` | The atlas's slot fixture: one toy block — qkv 768 columns, out 512, ffn1 2048, ffn2 as 4 K-blocks of 512 whose partials reduce — mapped onto 512 x 512 : 64 (mux 8) with 6 macros per chip. ffn1 does not fit the slots chip 0 has left, so it moves WHOLE to chip 1 and pays a crossing, and ffn2 does the same to chip 2: three chiplets, two crossings, 44 stranded slots, 30.6% cell waste, L = 69 ticks and 9 sequences in flight. It is EMITTED, not typed: a hand document drifts from the contract it claims to illustrate, and this one twice did. Rebuild it with `python tools/fws_toy_slotmap.py --fixture`; a test compares the bytes, so a contract change that is not carried into it fails. |
| `../dse/toy_slotmap.json` and `../dse/toy_slotmap_frontier/` | The real sweep the view is read on: `?bundle=toy_slotmap`. Twelve bundles over six platform points — mux 8, 16, 32, 64, 128 and one at Z = 512 where qkv is ragged. Every point carries its own tick schedule and its checker's verdict. |
