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
| `role` | string | `tp` \| `ep` \| `pp` \| `pd` \| `act`. |
| `bytes` | number | Bytes per `per`. |
| `per` | string | The unit `bytes` is counted over, e.g. `decode_step`, `inference`. |
| `label` | string | What the traffic is. |
| `basis` | string | The arithmetic behind `bytes`. |

`act` extends the plan page's four roles. It carries activation traffic that
is not a parallelism collective: the chip-boundary handoff, and the analog →
shared-digital hop that attention needs (D13). The alternative was to call a
chip boundary `pp`, which ADJ-5 forbids — chip index is not implicitly pp.

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
the atlas and press **Self-test**, or open it with `?selftest=1`. It runs 37
cases: the clean document plus one per corruption — one for every rule the
loader can emit, with extra cases for rules 1, 9, 18 and 24, which refuse
more than one way. `tests/test_qif_atlas.py` asserts that the integer in this
sentence equals `ATLAS.selfTest(doc).length`, so it cannot rot again. `tests/test_qif_atlas.py` runs the same self-test headlessly in
node, so a rule that stops firing fails the suite rather than a screenshot.

A corruption a given document gives the harness no way to BUILD — rule 13
needs a macro holding two tiles, rule 22 needs a second shared digital
chiplet — is reported as `n/a`, "not constructible on this document". It is
neither a pass nor a miss, and calling it a miss would read as a broken loader
on a perfectly good file.

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
| `?data=<file>` | Load a sibling document from the atlas's own directory instead of the embedded fixture. A bare basename; a path is ignored. Needs http (a `file://` page cannot fetch). Without it the page keeps the EMBEDDED document: the sibling fetch runs only when this parameter asks for one, or when the file carries no embedded blob at all, so an embedded document is never silently replaced by whatever else sits in the directory. A dropped or picked file always wins over an in-flight fetch. |
| `?view=package\|chip\|macro` | Open at that level. |
| `?chip=<chip id>`, `?macro=<macro id>` | Which chip or macro to open. A macro implies its own chip. |
| `?duty=1` | Turn the duty-cycle overlay on. |
| `?model=`, `?layer=`, `?group=`, `?pool=`, `?q=` | Pre-set the filters and the owner search. |
| `?selftest=1` | Open the self-test panel: refusals and view rules. |

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
| `fixture_llama7b_tp2.json` | The P5.1 phase-0 fixture: a hand-authored Llama2-7B tp = 2 decode placement with its invented fields marked. It is what `atlas.html` embeds and what the loader self-test is proved against. |
| `p3_llama7b_tp2.json` | The same model emitted by the real producer, placement only (P3.6). |
| `granite_4_0_h_tiny.json` | The ADJ-1 headline artifact: Granite-4.0-H-Tiny, placed and priced (`--priced`). |
| `pd_llama7b.json` | The PD pair (D16): a prefill inventory and a decode inventory with different layer packings and chiplet counts, one priced handoff, both halves priced on their OWN timeline. It is the two-system picture. |

Each exported document names in `provenance.reference_run.command` the exact
command that regenerates it, and each is pinned by a regenerate-and-compare
test — a stale artifact is a wrong picture under a right title.

## Keeping the embedded copy honest

`atlas.html` embeds a copy of the fixture in a
`<script type="application/json" id="atlas-embedded">` block so one file can
be handed to someone with its mapping already inside. When the fixture
changes, re-embed it — the header prints which source loaded and the
document's `provenance.authored` date, so a stale copy is visible rather than
silent.
