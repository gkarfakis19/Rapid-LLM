# Web UI Improvement Plan

This plan tracks the first ten highest-return Web UI improvements.

| Rank | Status | Improvement | Importance | Ease | Implementation target |
|---:|---|---|---|---|---|
| 1 | Done | Stabilize long-running actions | Very high | Medium | Avoid full content replacement during polling, preserve open panels, and keep progress updates scoped to status regions. |
| 2 | Done | Standardize modal/dialog styling | Very high | Easy | Make Details modal header, close affordance, outside-click, title sizing, body spacing, and toolbar layout consistent. |
| 3 | Done | Improve sweep builder hierarchy | Very high | Medium | Make each sweep dimension read as field, values, preview, and active targets with clearer sectioning. |
| 4 | Done | Add compact run summary before launch | High | Easy | Show model, hardware, mode, sweep size, workers, timeout, metric, and output shape near the launch button. |
| 5 | Done | Make error boxes specific and actionable | High | Easy | Show error titles, rejected fields/values where available, and concise next-step guidance. |
| 6 | Done | Contain the right rail | High | Easy | Use a sticky right rail with inner overflow so launch/status/progress panels cannot overlap main content. |
| 7 | Done | Polish plots | High | Medium | Keep explicit OOM styles, legends, axis labels, units, hover labels, and stable plot sizing. |
| 8 | Done | Add export/download affordances | High | Medium | Make Details exports visible as download actions and expose saved local paths as links where possible. |
| 9 | Done | Make sweep presets visibly reusable | Medium-high | Medium | Add visible preset-style controls for common sweep shapes without hiding expressivity. |
| 10 | Done | Add sweep preview chips | Medium-high | Easy | Show compact chips such as `Batch x3`, `Network BW Scale x3`, and `Model Config x2` for active dimensions. |

Verification:
- Focused service and formatting tests cover preview construction, Details rendering, exports, and plot styling.
- Browser smoke covers desktop, stacked, mobile, and Details modal views with screenshots.

Additional completed improvements:
- Keyboard and focus behavior for Details: Escape closes the dialog, the close button receives focus on open, and focus returns to the opener after close.
