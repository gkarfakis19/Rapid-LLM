from __future__ import annotations

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


@pytest.mark.webui_browser
def test_webui_layout_and_visual_health(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    port = _free_port()
    workspace = tmp_path / "workspace"
    env = {
        **os.environ,
        "RAPID_WEBUI_PORT": str(port),
        "RAPID_WEBUI_WORKSPACE_ROOT": str(workspace),
        "RAPID_WEBUI_PYTHON_BIN": str(Path(".venv/bin/python").resolve()),
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
            page.get_by_text("Launch Preview").wait_for(timeout=15000)
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
                        hasBuilder: text.includes('1 Run'),
                        hasHistory: text.includes('2 History'),
                        hasDetails: text.includes('3 Details'),
                        hasRunSetup: text.includes('Run Setup'),
                        hasBasicOptions: text.includes('Basic Options'),
                        hasHeaderHelp: text.includes('Run') && text.includes('History') && text.includes('Details') && text.includes('Hover any control for details.'),
                        hasEditorTabs: text.includes('Selected file editor') && text.includes('MODEL FILES') && text.includes('HARDWARE FILES'),
                        hasDespisedRunSetupCopy: text.includes('Choose one hardware target and any number of models'),
                        hasParallelismSearch: text.includes('Parallelism search'),
                        hasUseAstraSim: text.includes('Use AstraSim'),
                        hasAdvancedBackendMode: text.includes('Execution backend') || text.includes('Execution mode') || text.includes('hybrid') || text.includes('flattened'),
                        hasWorkers: text.includes('Workers') && text.includes('CPU cores detected'),
                        hasWorkflow: text.includes('Run -> History -> Load -> Details'),
                        hasRawOverride: text.includes('Use raw YAML override'),
                        hasReadOnlySeeds: text.includes('Read-only seeds'),
                        hasMetaReferences: /\\bv1\\b|prototype|deferred|future/i.test(text),
                        darkInputIconCount: Array.from(document.querySelectorAll('.mantine-Input-section svg, .mantine-Select-section svg, .mantine-MultiSelect-section svg, .mantine-NumberInput-section svg, .mantine-NumberInput-controls svg, .mantine-NumberInput-control svg'))
                            .filter((el) => {
                                const style = getComputedStyle(el);
                                const color = (style.stroke && style.stroke !== 'none' ? style.stroke : style.color).match(/\\d+/g);
                                if (!color) return false;
                                const [r, g, b] = color.map(Number);
                                return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 85;
                            }).length,
                        lowContrastTextCount: Array.from(document.querySelectorAll('.mantine-Alert-root, .mantine-Accordion-control'))
                            .filter((el) => {
                                const color = getComputedStyle(el).color.match(/\\d+/g);
                                if (!color) return false;
                                const [r, g, b] = color.map(Number);
                                return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 110;
                            }).length,
                    };
                }"""
            )
            assert desktop_metrics["scrollWidth"] <= desktop_metrics["clientWidth"] + 2
            assert desktop_metrics["paperCount"] >= 6
            assert desktop_metrics["buttonCount"] >= 6
            assert desktop_metrics["uniqueBackgrounds"] >= 6
            assert desktop_metrics["hasBuilder"]
            assert desktop_metrics["hasHistory"]
            assert desktop_metrics["hasDetails"]
            assert desktop_metrics["hasRunSetup"]
            assert desktop_metrics["hasBasicOptions"]
            assert desktop_metrics["hasHeaderHelp"]
            assert desktop_metrics["hasEditorTabs"]
            assert not desktop_metrics["hasDespisedRunSetupCopy"]
            assert desktop_metrics["hasParallelismSearch"]
            assert desktop_metrics["hasUseAstraSim"]
            assert not desktop_metrics["hasAdvancedBackendMode"]
            assert desktop_metrics["hasWorkers"]
            assert desktop_metrics["hasWorkflow"]
            assert not desktop_metrics["hasRawOverride"]
            assert not desktop_metrics["hasReadOnlySeeds"]
            assert not desktop_metrics["hasMetaReferences"]
            assert desktop_metrics["darkInputIconCount"] == 0
            assert desktop_metrics["lowContrastTextCount"] == 0
            page.get_by_role("tab", name="2 History").click()
            page.get_by_text("No runs yet. Use the builder tab to create one.").wait_for(timeout=5000)
            page.get_by_role("tab", name="1 Run").click()
            page.get_by_label("Batch size").hover()
            page.get_by_text("Global batch size across all participating devices.").wait_for(timeout=5000)
            page.get_by_label("Optimize parallelism").hover()
            page.get_by_text("Search TP, CP, PP, DP, and EP combinations for each Total GPUs target.").wait_for(timeout=5000)
            page.get_by_label("Use AstraSim").hover()
            page.get_by_text("Run the simulator through AstraSim using hierarchical mode.").wait_for(timeout=5000)
            page.get_by_label("Workers").hover()
            page.get_by_text("Number of local worker processes used inside a sweep.").wait_for(timeout=5000)
            page.get_by_text("Selected file editor").hover()
            page.get_by_text("Switch tabs before changing fields to edit a different selected file.").wait_for(timeout=5000)
            page.locator("#adv-model-mode").scroll_into_view_if_needed()
            page.get_by_text("Advanced Options").wait_for(timeout=5000)
            page.get_by_text("Model").first.wait_for(timeout=5000)
            page.get_by_text("Hardware").first.wait_for(timeout=5000)
            page.get_by_text("Type guide").wait_for(timeout=5000)
            page.get_by_text("LLM").first.wait_for(timeout=5000)
            page.get_by_text("ViT").first.wait_for(timeout=5000)
            page.get_by_text("GEMM").first.wait_for(timeout=5000)
            page.locator("#adv-model-mode").hover()
            page.get_by_text("Select the model execution family written to model_param.mode.").wait_for(timeout=5000)
            page.get_by_text("ViT").first.hover()
            page.get_by_text("Vision Transformer path.").wait_for(timeout=5000)

            page.locator("#dim-1-field").scroll_into_view_if_needed()
            page.locator("#dim-1-field").click(force=True)
            sweep_option_text = page.evaluate(
                """() => Array.from(document.querySelectorAll('.mantine-Combobox-option, .mantine-Select-option, [role="option"]')).map((el) => el.textContent || '').join('\\n')"""
            )
            assert "Tensor Parallelism" not in sweep_option_text
            assert "Data Parallelism" not in sweep_option_text
            assert "Total GPUs" in sweep_option_text
            page.get_by_role("option", name="Batch Size").click()
            page.wait_for_timeout(250)
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
            page.wait_for_timeout(250)
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
            page.locator("#run-mode").get_by_text("Single Run").click()
            page.wait_for_timeout(250)
            single_run_state = page.evaluate(
                """() => ({
                    sweepDisplay: getComputedStyle(document.querySelector('#sweep-dimensions-section')).display,
                    hasPreviewButton: document.body.innerText.includes('Preview Launch'),
                    hasRunButton: document.body.innerText.includes('Run'),
                    visibleSweepTitle: Array.from(document.querySelectorAll('*')).some((el) => {
                        if (el.textContent !== 'Sweep Dimensions') return false;
                        return el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
                    }),
                })"""
            )
            assert single_run_state["sweepDisplay"] == "none"
            assert single_run_state["hasPreviewButton"]
            assert single_run_state["hasRunButton"]
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
