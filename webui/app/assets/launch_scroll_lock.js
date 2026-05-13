(function () {
  "use strict";

  if (window.rapidLaunchScrollLockInstalled) {
    return;
  }
  window.rapidLaunchScrollLockInstalled = true;

  var LOCK_CLASS = "rapid-launch-scroll-lock";
  var DESKTOP_QUERY = "(min-width: 75.01em)";
  var desktopMedia = window.matchMedia(DESKTOP_QUERY);
  function builderGrid() {
    return document.querySelector(".builder-grid");
  }

  function builderIsVisible() {
    var grid = builderGrid();
    if (!grid) {
      return false;
    }
    var style = window.getComputedStyle(grid);
    if (style.display === "none" || style.visibility === "hidden") {
      return false;
    }
    return Boolean(grid && grid.getClientRects().length && grid.offsetWidth && grid.offsetHeight);
  }

  function modalIsVisible() {
    var overlay = document.getElementById("detail-overlay");
    if (!overlay) {
      return false;
    }
    var style = window.getComputedStyle(overlay);
    return style.display !== "none" && style.visibility !== "hidden";
  }

  function lockIsActive() {
    return desktopMedia.matches && builderIsVisible() && !modalIsVisible();
  }

  function setLaunchPanelHeight() {
    var grid = builderGrid();
    var left = leftScroller();
    if (!grid || !desktopMedia.matches) {
      document.documentElement.style.removeProperty("--launch-panel-height");
      document.documentElement.style.removeProperty("--launch-left-scroll-height");
      if (left) {
        left.style.removeProperty("height");
        left.style.removeProperty("max-height");
        left.style.removeProperty("overflow-y");
      }
      return;
    }
    var rect = grid.getBoundingClientRect();
    var available = Math.max(420, window.innerHeight - rect.top - 12);
    document.documentElement.style.setProperty("--launch-panel-height", available + "px");
    document.documentElement.style.setProperty("--launch-left-scroll-height", available + "px");
    if (left) {
      left.style.height = available + "px";
      left.style.maxHeight = available + "px";
      left.style.overflowY = "scroll";
    }
  }

  function syncLockClass() {
    var active = lockIsActive();
    document.documentElement.classList.toggle(LOCK_CLASS, active);
    document.body.classList.toggle(LOCK_CLASS, active);
    setLaunchPanelHeight();
    if (active && (window.scrollX || window.scrollY)) {
      window.scrollTo(0, 0);
    }
  }

  function leftScroller() {
    return document.querySelector(".builder-left-scroll");
  }

  function wheelDeltaPixels(event, scroller) {
    var delta = event.deltaY || event.deltaX || 0;
    if (event.deltaMode === 1) {
      return delta * 16;
    }
    if (event.deltaMode === 2) {
      return delta * (scroller ? scroller.clientHeight : window.innerHeight);
    }
    return delta;
  }

  function shouldKeepNativeScroll(target) {
    if (!(target instanceof Element)) {
      return false;
    }
    return Boolean(
      target.closest(
        [
          "#detail-overlay",
          ".mantine-Select-dropdown",
          ".mantine-MultiSelect-dropdown",
          ".mantine-Combobox-dropdown",
          ".mantine-Popover-dropdown",
          ".mantine-Menu-dropdown",
          ".mantine-Modal-root",
          "[role='listbox']",
          "textarea",
          "[contenteditable='true']",
        ].join(",")
      )
    );
  }

  function redirectWheel(event) {
    if (!lockIsActive() || shouldKeepNativeScroll(event.target)) {
      return;
    }
    var scroller = leftScroller();
    if (!scroller) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    var before = scroller.scrollTop;
    scroller.scrollTop = before + wheelDeltaPixels(event, scroller);
  }

  function start() {
    syncLockClass();
    window.addEventListener("resize", syncLockClass, { passive: true });
    window.addEventListener("orientationchange", syncLockClass, { passive: true });
    document.addEventListener("wheel", redirectWheel, { capture: true, passive: false });
    document.addEventListener("click", function () {
      window.setTimeout(syncLockClass, 0);
    }, true);
    document.addEventListener("keydown", function () {
      window.setTimeout(syncLockClass, 0);
    }, true);

    var observer = new MutationObserver(syncLockClass);
    observer.observe(document.body, {
      attributes: true,
      childList: true,
      subtree: true,
      attributeFilter: ["class", "style", "aria-hidden", "data-active"],
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
