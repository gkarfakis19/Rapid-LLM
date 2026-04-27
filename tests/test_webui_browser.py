from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pytest


def _free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError as exc:
        pytest.skip(f"Local socket creation is blocked in this environment: {exc}")


def _wait_for_http(url: str, timeout_s: float = 25.0) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def _launch_browser(playwright):
    attempts = [("chromium", playwright.chromium, None)]
    firefox_path = shutil.which("firefox")
    attempts.append(("firefox", playwright.firefox, firefox_path))
    errors = []
    for name, browser_type, executable_path in attempts:
        kwargs = {"headless": True}
        if executable_path:
            kwargs["executable_path"] = executable_path
        try:
            return browser_type.launch(**kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    pytest.skip("No Playwright-compatible browser available: " + " | ".join(errors))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_completed_sweep(workspace: Path) -> None:
    now = "2026-04-26T20:00:00+00:00"
    sweep_root = workspace / "sweeps" / "sweep-ui-detail"
    payload = {
        "run_mode": "sweep",
        "model_preset_id": "Llama2-7B.yaml",
        "hardware_preset_id": "A100_SXM4_80GB.yaml",
        "metric": "training_time_s",
        "x_axis": "batch_size",
        "series_axis": None,
        "simple": {"run_type": "training", "tp": 4, "cp": 1, "pp": 1, "dp": 1, "ep": 1, "replica_count": 1},
    }
    _write_json(
        sweep_root / "request.json",
        {
            "id": "sweep-ui-detail",
            "kind": "sweep",
            "created_at": now,
            "title": "Detail Smoke",
            "payload": payload,
            "preview": {"optimizer_enabled": False, "top_level_case_count": 2},
        },
    )
    _write_json(sweep_root / "status.json", {"created_at": now, "updated_at": now, "status": "completed", "progress_completed": 2, "progress_total": 2})
    _write_json(sweep_root / "summary.json", {"title": "Detail Smoke", "best_metric_label": "Training Time", "best_metric_value": 1.2})
    _write_json(
        sweep_root / "cases" / "case-0001.json",
        {
            "case_id": "case-0001",
            "label": "batch 64",
            "status": "completed",
            "dimension_values": {"batch_size": 64, "model_config": "Llama2-7B.yaml", "hardware_config": "A100_SXM4_80GB.yaml"},
            "metrics": {
                "training_time_s": 1.2,
                "total_flops": 4.56e14,
                "achieved_flops": 3.2e14,
                "peak_flops_per_gpu": 3.9e14,
                "peak_system_flops": 1.56e15,
                "num_gpus": 4,
                "memory_exceeded": False,
                "memory_violation_gb": 0.0,
            },
        },
    )
    _write_json(
        sweep_root / "cases" / "case-0002.json",
        {
            "case_id": "case-0002",
            "label": "batch 128",
            "status": "failed",
            "dimension_values": {"batch_size": 128, "model_config": "Llama2-7B.yaml", "hardware_config": "A100_SXM4_80GB.yaml"},
            "metrics": {
                "training_time_s": 9.9,
                "total_flops": 9.12e14,
                "achieved_flops": 6.4e14,
                "peak_flops_per_gpu": 3.9e14,
                "peak_system_flops": 1.56e15,
                "num_gpus": 4,
                "memory_exceeded": True,
                "memory_violation_gb": 12.5,
            },
        },
    )


@pytest.mark.webui_browser
def test_webui_layout_and_visual_health(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    port = _free_port()
    workspace = tmp_path / "workspace"
    _seed_completed_sweep(workspace)
    env = {
        **os.environ,
        "RAPID_WEBUI_PORT": str(port),
        "RAPID_WEBUI_WORKSPACE_ROOT": str(workspace),
        "RAPID_WEBUI_PYTHON_BIN": ".venv/bin/python",
        "PYTHONUNBUFFERED": "1",
    }
    stdout = tmp_path / "webui.stdout.log"
    stderr = tmp_path / "webui.stderr.log"
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(
            [sys.executable, "-m", "webui.app.main"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=out,
            stderr=err,
        )
    try:
        url = f"http://127.0.0.1:{port}"
        _wait_for_http(url)
        with sync_api.sync_playwright() as playwright:
            browser = _launch_browser(playwright)
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            console_errors: list[str] = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.goto(url, wait_until="networkidle")
            page.get_by_text("RAPID-LLM Workbench").wait_for(timeout=15000)
            page.get_by_text("Launch Plan").wait_for(timeout=15000)
            screenshot_dir = Path(os.environ.get("RAPID_WEBUI_SCREENSHOT_DIR", tmp_path))
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            page.locator("#model-run-configs").click(force=True)
            page.locator(".mantine-Combobox-dropdown, .mantine-Select-dropdown, .mantine-MultiSelect-dropdown, .mantine-Popover-dropdown").first.wait_for(timeout=5000)
            dropdown_metrics = page.evaluate(
                """() => {
                    const dropdowns = Array.from(document.querySelectorAll('.mantine-Combobox-dropdown, .mantine-Select-dropdown, .mantine-MultiSelect-dropdown, .mantine-Popover-dropdown'))
                        .filter((el) => el.getAttribute('data-hidden') !== 'true');
                    const dropdown = dropdowns[0];
                    const options = dropdown
                        ? Array.from(dropdown.querySelectorAll('.mantine-Combobox-option, .mantine-Select-option, [role="option"]'))
                        : [];
                    return {
                        optionCount: options.length,
                        lowContrastOptions: options.filter((el) => {
                            const style = getComputedStyle(el);
                            const color = style.color.match(/\\d+/g);
                            const bg = getComputedStyle(dropdown).backgroundColor.match(/\\d+/g);
                            if (!color || !bg) return true;
                            const [r, g, b] = color.map(Number);
                            const [br, bgc, bb] = bg.map(Number);
                            const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b);
                            const bgLum = (0.2126 * br + 0.7152 * bgc + 0.0722 * bb);
                            return Math.abs(lum - bgLum) < 95;
                        }).length,
                    };
                }"""
            )
            assert dropdown_metrics["optionCount"] >= 1
            assert dropdown_metrics["lowContrastOptions"] == 0
            page.screenshot(path=screenshot_dir / f"webui-dropdown-{stamp}.png", full_page=True)
            page.keyboard.press("Escape")
            page.evaluate("document.querySelector('.builder-left-scroll')?.scrollTo(0, 0)")

            page.screenshot(path=screenshot_dir / f"webui-desktop-{stamp}.png", full_page=True)

            desktop_metrics = page.evaluate(
                """() => {
                    const doc = document.documentElement;
                    const text = document.body.innerText;
                    const backgrounds = Array.from(document.querySelectorAll('*'))
                        .map((el) => getComputedStyle(el).backgroundColor)
                        .filter((color) => color && color !== 'rgba(0, 0, 0, 0)');
                    return {
                        scrollWidth: doc.scrollWidth,
                        clientWidth: doc.clientWidth,
                        paperCount: document.querySelectorAll('.mantine-Paper-root').length,
                        buttonCount: document.querySelectorAll('button').length,
                        uniqueBackgrounds: Array.from(new Set(backgrounds)).length,
                        hasBuilder: text.includes('1 Launch'),
                        hasHistory: text.includes('2 Run log'),
                        hasTopDetailsTab: text.includes('3 Details'),
                        hasOldLoadDetailsButton: text.includes('Load Details'),
                        hasLaunchSetup: text.includes('Launch Setup'),
                        hasConfigOptions: text.includes('Config Options'),
                        hasBasicOptions: text.includes('Basic Options'),
                        hasAdvancedOptions: text.includes('Advanced Options'),
                        hasHeaderHelp: text.includes('Launch') && text.includes('Run log') && text.includes('Details') && text.includes('Hover any control for details.'),
                        hasNanocadLogo: !!document.querySelector('img.nanocad-logo[src*="nanocad-logo.png"][alt="NanoCAD"]'),
                        hasNanocadLogoFrame: !!document.querySelector('.nanocad-logo-frame img.nanocad-logo[src*="nanocad-logo.png"]'),
                        hasBrandOrb: !!document.querySelector('.brand-orb'),
                        topbarTitleColor: getComputedStyle(document.querySelector('.topbar-title')).color,
                        hoverHelpColor: getComputedStyle(document.querySelector('.flow-hover-copy')).color,
                        telemetryBackdrop: getComputedStyle(document.querySelector('.telemetry-pills')).backgroundColor,
                        hasSettingsBadge: text.includes('Updates as settings change'),
                        hasRuntimeScalingCopy: text.includes('rapidly approach worst-case') && text.includes('beyond 256 GPUs'),
                        hasEditorTabs: text.includes('ACTIVE YAML WORKBOOK') && text.includes('Model') && text.includes('Hardware'),
                        hasFileActions: text.includes('Active file actions') && text.includes('New copy') && text.includes('Rename'),
                        hasDespisedRunSetupCopy: text.includes('Choose one hardware target and any number of models'),
                        hasParallelismSearch: text.includes('Parallelism search'),
                        hasOptimizerWarning: text.includes('WARNING: This may increase runtime dramatically.'),
                        hasUseAstraSim: text.includes('Use AstraSim'),
                        hasAdvancedBackendMode: text.includes('Execution backend') || text.includes('Execution mode') || text.includes('hybrid') || text.includes('flattened'),
                        hasWorkers: text.includes('Workers') && text.includes('CPU cores detected'),
                        hasWorkflowCaption: text.includes('Launch -> Run log -> Details'),
                        hasPreviewButton: text.includes('Preview Launch'),
                        hasLiveLaunchButton: /Launch \\d+ runs?/.test(text),
                        hasRawOverride: text.includes('Use raw YAML override'),
                        hasReadOnlySeeds: text.includes('Read-only seeds'),
                        hasMetaReferences: /\\bv1\\b|prototype|deferred|future/i.test(text),
                        lowContrastIconCount: (() => {
                            const rgb = (raw) => {
                                const nums = (raw || '').match(/\\d+(?:\\.\\d+)?/g);
                                if (nums && nums.length >= 4 && Number(nums[3]) === 0) return null;
                                return nums ? nums.slice(0, 3).map(Number) : null;
                            };
                            const channel = (v) => {
                                const s = v / 255;
                                return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
                            };
                            const lum = (c) => 0.2126 * channel(c[0]) + 0.7152 * channel(c[1]) + 0.0722 * channel(c[2]);
                            const ratio = (a, b) => {
                                const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
                                return (hi + 0.05) / (lo + 0.05);
                            };
                            const bgFor = (el) => {
                                let cur = el;
                                while (cur && cur !== document.documentElement) {
                                    const bg = rgb(getComputedStyle(cur).backgroundColor);
                                    if (bg) return bg;
                                    cur = cur.parentElement;
                                }
                                return [255, 255, 255];
                            };
                            return Array.from(document.querySelectorAll('.mantine-Input-section svg, .mantine-Select-section svg, .mantine-MultiSelect-section svg, .mantine-NumberInput-section svg, .mantine-NumberInput-controls svg, .mantine-NumberInput-control svg'))
                                .filter((el) => {
                                    const style = getComputedStyle(el);
                                    const color = rgb(style.stroke && style.stroke !== 'none' ? style.stroke : style.color);
                                    return color && ratio(color, bgFor(el)) < 3;
                                }).length;
                        })(),
                        lowContrastTextSamples: (() => {
                            const rgb = (raw) => {
                                const nums = (raw || '').match(/\\d+(?:\\.\\d+)?/g);
                                if (nums && nums.length >= 4 && Number(nums[3]) === 0) return null;
                                return nums ? nums.slice(0, 3).map(Number) : null;
                            };
                            const channel = (v) => {
                                const s = v / 255;
                                return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
                            };
                            const lum = (c) => 0.2126 * channel(c[0]) + 0.7152 * channel(c[1]) + 0.0722 * channel(c[2]);
                            const ratio = (a, b) => {
                                const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
                                return (hi + 0.05) / (lo + 0.05);
                            };
                            const bgFor = (el) => {
                                let cur = el;
                                while (cur && cur !== document.documentElement) {
                                    const bg = rgb(getComputedStyle(cur).backgroundColor);
                                    if (bg) return bg;
                                    cur = cur.parentElement;
                                }
                                return [255, 255, 255];
                            };
                            return Array.from(document.querySelectorAll('.mantine-Alert-root, .mantine-Accordion-control, .mantine-Input-input, .mantine-Select-input, .mantine-NumberInput-input, .mantine-Tabs-tab, button'))
                                .filter((el) => el.getClientRects().length > 0)
                                .map((el) => {
                                    const text = (el.textContent || el.value || '').trim();
                                    if (!text) return null;
                                    const color = rgb(getComputedStyle(el).color);
                                    if (!color) return null;
                                    const bg = bgFor(el);
                                    const contrast = ratio(color, bg);
                                    return contrast < 4.5 ? {text: text.slice(0, 40), contrast} : null;
                                })
                                .filter(Boolean)
                                .slice(0, 8);
                        })(),
                    };
                }"""
            )
            assert desktop_metrics["scrollWidth"] <= desktop_metrics["clientWidth"] + 2
            assert desktop_metrics["paperCount"] >= 6
            assert desktop_metrics["buttonCount"] >= 6
            assert desktop_metrics["uniqueBackgrounds"] >= 6
            assert desktop_metrics["hasBuilder"]
            assert desktop_metrics["hasHistory"]
            assert not desktop_metrics["hasTopDetailsTab"]
            assert desktop_metrics["hasOldLoadDetailsButton"] is False
            assert desktop_metrics["hasLaunchSetup"]
            assert desktop_metrics["hasConfigOptions"]
            assert desktop_metrics["hasBasicOptions"]
            assert desktop_metrics["hasAdvancedOptions"]
            assert desktop_metrics["hasHeaderHelp"]
            assert desktop_metrics["hasNanocadLogo"]
            assert desktop_metrics["hasNanocadLogoFrame"]
            assert not desktop_metrics["hasBrandOrb"]
            assert desktop_metrics["topbarTitleColor"] == "rgb(255, 255, 255)"
            assert desktop_metrics["hoverHelpColor"] == "rgb(223, 245, 253)"
            assert desktop_metrics["telemetryBackdrop"] != "rgba(0, 0, 0, 0)"
            assert not desktop_metrics["hasSettingsBadge"]
            assert desktop_metrics["hasRuntimeScalingCopy"]
            assert desktop_metrics["hasEditorTabs"]
            assert desktop_metrics["hasFileActions"]
            assert not desktop_metrics["hasDespisedRunSetupCopy"]
            assert not desktop_metrics["hasParallelismSearch"]
            assert not desktop_metrics["hasOptimizerWarning"]
            assert not desktop_metrics["hasUseAstraSim"]
            assert not desktop_metrics["hasAdvancedBackendMode"]
            assert desktop_metrics["hasWorkers"]
            assert not desktop_metrics["hasWorkflowCaption"]
            assert not desktop_metrics["hasPreviewButton"]
            assert desktop_metrics["hasLiveLaunchButton"]
            assert not desktop_metrics["hasRawOverride"]
            assert not desktop_metrics["hasReadOnlySeeds"]
            assert not desktop_metrics["hasMetaReferences"]
            assert desktop_metrics["lowContrastIconCount"] == 0
            assert desktop_metrics["lowContrastTextSamples"] == []

            page.set_viewport_size({"width": 1024, "height": 900})
            page.get_by_text("Config Options", exact=True).scroll_into_view_if_needed()
            page.screenshot(path=screenshot_dir / f"webui-stacked-config-options-{stamp}.png", full_page=True)
            stacked_config_metrics = page.evaluate(
                """() => {
                    const visible = (el) => !!el && el.getClientRects().length > 0 && getComputedStyle(el).display !== 'none' && getComputedStyle(el).visibility !== 'hidden';
                    const grid = document.querySelector('.builder-grid');
                    const left = document.querySelector('.builder-left-scroll');
                    const card = document.querySelector('.config-options-card');
                    const tabs = document.querySelector('.config-workbook-tab-list');
                    const actions = document.querySelector('.config-file-actions');
                    const gridStyle = grid ? getComputedStyle(grid) : null;
                    const leftStyle = left ? getComputedStyle(left) : null;
                    const cardStyle = card ? getComputedStyle(card) : null;
                    const gridRect = grid ? grid.getBoundingClientRect() : {bottom: 0};
                    const cardRect = card ? card.getBoundingClientRect() : {bottom: 0, height: 0};
                    return {
                        gridOverflowY: gridStyle ? gridStyle.overflowY : null,
                        gridHeight: gridStyle ? gridStyle.height : null,
                        leftOverflowY: leftStyle ? leftStyle.overflowY : null,
                        leftMaxHeight: leftStyle ? leftStyle.maxHeight : null,
                        cardOverflowY: cardStyle ? cardStyle.overflowY : null,
                        cardHeight: cardRect.height,
                        cardClippedByGrid: !!(grid && card && gridStyle.overflowY !== 'visible' && cardRect.bottom > gridRect.bottom + 2),
                        tabsVisible: visible(tabs),
                        actionsVisible: visible(actions),
                    };
                }"""
            )
            assert stacked_config_metrics["gridOverflowY"] == "visible"
            assert stacked_config_metrics["leftOverflowY"] == "visible"
            assert stacked_config_metrics["cardOverflowY"] == "visible"
            assert not stacked_config_metrics["cardClippedByGrid"]
            assert stacked_config_metrics["tabsVisible"]
            assert stacked_config_metrics["actionsVisible"]

            page.get_by_role("tab", name="2 Run log").click()
            page.get_by_text("Detail Smoke").wait_for(timeout=5000)
            page.locator("button").filter(has_text="Details").first.click()
            page.get_by_role("dialog").wait_for(timeout=5000)
            page.locator("#detail-plot-toolbar").get_by_text("Line Plot", exact=True).wait_for(timeout=5000)
            page.locator("#detail-plot-toolbar").get_by_text("Scatter", exact=True).wait_for(timeout=5000)
            page.locator("#detail-plot-toolbar").get_by_text("Bar chart", exact=True).wait_for(timeout=5000)
            page.locator("#detail-plot-toolbar").get_by_text("Save plot", exact=True).wait_for(timeout=5000)
            page.locator("#detail-plot-toolbar").get_by_text("Save plot", exact=True).click()
            page.get_by_text("Saved plot:").wait_for(timeout=5000)
            saved_plot_text = page.locator("#plot-save-status").inner_text(timeout=5000)
            assert saved_plot_text.endswith(".png")
            saved_plot_path = Path(saved_plot_text.removeprefix("Saved plot: ").strip())
            assert saved_plot_path.exists()
            assert saved_plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert saved_plot_path.stat().st_size > 20_000
            page.get_by_text("Memory Exceeded").wait_for(timeout=5000)
            page.get_by_text("12.5 GB").wait_for(timeout=5000)
            page.get_by_text("case-0001 - Llama2-7B").wait_for(timeout=5000)
            page.get_by_text("Fixed parallelism").wait_for(timeout=5000)
            page.get_by_text("Early termination rate: 50.0% (1/2 runs).").wait_for(timeout=5000)
            page.get_by_text("Many runs terminated early").wait_for(timeout=5000)
            page.get_by_text("0.00 us").wait_for(timeout=5000)
            filter_metrics = page.evaluate(
                """() => {
                    const input = Array.from(document.querySelectorAll('.dash-filter input, .dash-filter textarea'))
                        .find((el) => el.getClientRects().length > 0);
                    const style = input ? getComputedStyle(input) : null;
                    const placeholder = input ? getComputedStyle(input, '::placeholder') : null;
                    return {
                        found: !!input,
                        color: style?.color,
                        backgroundColor: style?.backgroundColor,
                        placeholderColor: placeholder?.color,
                    };
                }"""
            )
            assert filter_metrics["found"]
            assert filter_metrics["color"] == "rgb(23, 33, 43)"
            assert filter_metrics["backgroundColor"] == "rgb(255, 255, 255)"
            assert filter_metrics["placeholderColor"] != "rgb(255, 255, 255)"
            page.wait_for_function(
                """() => Array.from(document.querySelectorAll('.js-plotly-plot'))
                    .filter((plot) => plot.getClientRects().length > 0)
                    .some((plot) => ((plot.data || plot._fullData || []).length > 0))""",
                timeout=5000,
            )
            detail_metrics = page.evaluate(
                """() => {
                    const dialog = document.querySelector('[role="dialog"]');
                    const dialogText = dialog?.innerText || '';
                    const plot = Array.from(document.querySelectorAll('.js-plotly-plot')).find((item) => item.getClientRects().length > 0);
                    const legendText = Array.from(document.querySelectorAll('.legendtext')).map((el) => el.textContent || '').join('\\n');
                    return {
                        visible: !!dialog && getComputedStyle(dialog).display !== 'none',
                        hasStatusLegend: /status/i.test(legendText),
                        hasPlot: !!plot,
                        hasPeakGpuFlops: dialogText.includes('Peak FLOPS / GPU'),
                        hasPeakSystemFlops: dialogText.includes('Peak System FLOPS'),
                    };
                }"""
            )
            assert detail_metrics["visible"]
            assert detail_metrics["hasPlot"]
            assert not detail_metrics["hasStatusLegend"]
            assert not detail_metrics["hasPeakGpuFlops"]
            assert not detail_metrics["hasPeakSystemFlops"]
            page.screenshot(path=screenshot_dir / f"webui-details-{stamp}.png", full_page=True)
            page.locator("#detail-close-button").click()
            page.wait_for_function("() => getComputedStyle(document.querySelector('#detail-overlay')).display === 'none'", timeout=5000)
            page.get_by_role("tab", name="1 Launch").click()
            page.get_by_label("Batch size").hover()
            page.get_by_text("Global batch size across all participating devices.").wait_for(timeout=5000)
            page.locator("#simple-run-type").click(force=True)
            page.get_by_role("option", name="Inference").click()
            page.get_by_role("tab").filter(has_text="H100").click()
            page.get_by_text("Parallelism search").wait_for(timeout=5000)
            page.get_by_label("Optimize parallelism").hover()
            page.get_by_text("Search TP, CP, PP, DP, and EP combinations for each Total GPUs target.").wait_for(timeout=5000)
            page.wait_for_function("() => document.querySelector('#simple-tp-wrap')?.classList.contains('is-auto')", timeout=5000)
            auto_state = page.evaluate(
                """() => {
                    const wrap = document.querySelector('#simple-tp-wrap');
                    const input = document.querySelector('#simple-tp');
                    const replicaWrap = document.querySelector('#simple-replica-count-wrap');
                    const replicaInput = document.querySelector('#simple-replica-count');
                    return {
                        hasAutoClass: wrap.classList.contains('is-auto'),
                        autoContent: getComputedStyle(wrap, '::after').content,
                        inputTransparent: getComputedStyle(input).color === 'rgba(0, 0, 0, 0)' || getComputedStyle(input).color === 'transparent',
                        replicaHasAutoClass: replicaWrap.classList.contains('is-auto'),
                        replicaDisabled: replicaInput.disabled,
                        replicaInputTransparent: getComputedStyle(replicaInput).color === 'rgba(0, 0, 0, 0)' || getComputedStyle(replicaInput).color === 'transparent',
                    };
                }"""
            )
            assert auto_state["hasAutoClass"]
            assert auto_state["autoContent"] == '"Auto"'
            assert not auto_state["replicaHasAutoClass"]
            assert not auto_state["replicaDisabled"]
            assert not auto_state["replicaInputTransparent"]
            page.get_by_label("Use AstraSim").hover()
            page.get_by_text("Run the simulator through AstraSim using hierarchical mode.").wait_for(timeout=5000)
            page.get_by_label("Workers").hover()
            page.get_by_text("Number of worker processes used inside a sweep.").wait_for(timeout=5000)
            page.get_by_text("Config Options").hover()
            page.get_by_text("Choose the YAML file to edit.").wait_for(timeout=5000)
            page.get_by_role("tab").filter(has_text="Llama2").click()
            page.locator("#adv-model-type").scroll_into_view_if_needed()
            page.locator("#model-options-pane").get_by_text("Advanced Options").wait_for(timeout=5000)
            page.get_by_text("Model type guide").wait_for(timeout=5000)
            page.get_by_text("Hardware", exact=True).first.wait_for(timeout=5000)
            page.get_by_text("Execution family guide").wait_for(timeout=5000)
            page.get_by_text("deepseek_v3").first.wait_for(timeout=5000)
            page.get_by_text("glm4_moe").first.wait_for(timeout=5000)
            page.get_by_text("LLM").first.wait_for(timeout=5000)
            page.get_by_text("ViT").first.wait_for(timeout=5000)
            assert not page.get_by_text("GEMM").first.is_visible()
            page.locator("#adv-model-type").hover()
            page.get_by_text("Select the architecture family written to model_param.model_type.").wait_for(timeout=5000)
            page.get_by_text("deepseek_v3").first.hover()
            page.get_by_text("DeepSeek-V3 family.").wait_for(timeout=5000)
            page.locator("#adv-model-mode").hover()
            page.get_by_text("Select the execution family written to model_param.mode.").wait_for(timeout=5000)
            page.get_by_role("tab").filter(has_text="H100").click()
            page.locator("#adv-tensor-format").scroll_into_view_if_needed()
            page.locator("#adv-tensor-format").click(force=True)
            page.get_by_role("option", name="MXFP4 (4.25 bits)").wait_for(timeout=5000)
            tensor_option_text = page.evaluate(
                """() => Array.from(document.querySelectorAll('.mantine-Combobox-option, .mantine-Select-option, [role="option"]'))
                    .map((el) => el.textContent || '')
                    .join('\\n')"""
            )
            assert "MXFP4 (4.25 bits)" in tensor_option_text
            assert "INT4 (4 bits)" in tensor_option_text
            assert "FP8 (8 bits)" in tensor_option_text
            assert "FP32 (32 bits)" in tensor_option_text
            page.keyboard.press("Escape")

            page.locator("#dim-1-field").scroll_into_view_if_needed()
            page.locator("#dim-1-field").click(force=True)
            sweep_option_text = page.evaluate(
                """() => Array.from(document.querySelectorAll('.mantine-Combobox-option, .mantine-Select-option, [role="option"]')).map((el) => el.textContent || '').join('\\n')"""
            )
            assert "Tensor Parallelism" not in sweep_option_text
            assert "Data Parallelism" not in sweep_option_text
            assert "Total GPUs" in sweep_option_text
            page.get_by_role("option", name="Batch Size").click()
            page.wait_for_function("() => getComputedStyle(document.querySelector('#dim-1-values-wrap')).display !== 'none'", timeout=5000)
            sweep_values_state = page.evaluate(
                """() => ({
                    values: getComputedStyle(document.querySelector('#dim-1-values-wrap')).display,
                    configs: getComputedStyle(document.querySelector('#dim-1-configs-wrap')).display,
                    range: getComputedStyle(document.querySelector('#dim-1-range-wrap')).display,
                })"""
            )
            assert sweep_values_state["values"] != "none"
            assert sweep_values_state["configs"] == "none"
            assert sweep_values_state["range"] == "none"
            page.locator("#dim-1-mode-wrap").get_by_text("Range").click()
            page.wait_for_function("() => getComputedStyle(document.querySelector('#dim-1-range-wrap')).display !== 'none'", timeout=5000)
            sweep_range_state = page.evaluate(
                """() => ({
                    values: getComputedStyle(document.querySelector('#dim-1-values-wrap')).display,
                    configs: getComputedStyle(document.querySelector('#dim-1-configs-wrap')).display,
                    range: getComputedStyle(document.querySelector('#dim-1-range-wrap')).display,
                    stepLabelVisible: document.body.innerText.includes('Step size'),
                })"""
            )
            assert sweep_range_state["values"] == "none"
            assert sweep_range_state["configs"] == "none"
            assert sweep_range_state["range"] != "none"
            assert sweep_range_state["stepLabelVisible"]
            page.locator("#dim-1-start").fill("1")
            page.locator("#dim-1-end").fill("20")
            page.locator("#dim-1-step_or_points").fill("2")
            page.get_by_text("Preview: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 (10 values)").wait_for(timeout=5000)
            page.locator("#dim-1-end").fill("80")
            page.get_by_text("Preview: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, ... (40 values)").wait_for(timeout=5000)
            page.locator("#run-mode").get_by_text("Single Launch").click()
            page.wait_for_timeout(250)
            single_run_state = page.evaluate(
                """() => ({
                    sweepDisplay: getComputedStyle(document.querySelector('#sweep-dimensions-section')).display,
                    hasPreviewButton: document.body.innerText.includes('Preview Launch'),
                    hasLaunchButton: /Launch \\d+ runs?/.test(document.body.innerText),
                    visibleSweepTitle: Array.from(document.querySelectorAll('*')).some((el) => {
                        if (el.textContent !== 'Sweep Dimensions') return false;
                        return el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
                    }),
                })"""
            )
            assert single_run_state["sweepDisplay"] == "none"
            assert not single_run_state["hasPreviewButton"]
            assert single_run_state["hasLaunchButton"]
            assert not single_run_state["visibleSweepTitle"]

            page.evaluate("window.scrollTo(0, 0)")
            page.set_viewport_size({"width": 390, "height": 900})
            page.screenshot(path=screenshot_dir / f"webui-mobile-{stamp}.png", full_page=True)
            mobile_metrics = page.evaluate(
                """() => {
                    const doc = document.documentElement;
                    const tabs = Array.from(document.querySelectorAll('[role="tab"]')).map((el) => el.getBoundingClientRect());
                    return {
                        scrollWidth: doc.scrollWidth,
                        clientWidth: doc.clientWidth,
                        minTabWidth: Math.min(...tabs.map((rect) => rect.width)),
                    };
                }"""
            )
            assert mobile_metrics["scrollWidth"] <= mobile_metrics["clientWidth"] + 2
            assert mobile_metrics["minTabWidth"] >= 90
            assert not console_errors
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
