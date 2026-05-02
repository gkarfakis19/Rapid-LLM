(function () {
  "use strict";

  var OVERLAY_ID = "detail-overlay";
  var CLOSE_ID = "detail-close-button";
  var previousFocus = null;
  var wasVisible = false;
  var boundOverlay = null;

  function overlayElement() {
    return document.getElementById(OVERLAY_ID);
  }

  function modalIsVisible() {
    var overlay = overlayElement();
    if (!overlay) {
      return false;
    }
    var style = window.getComputedStyle(overlay);
    return style.display !== "none" && style.visibility !== "hidden";
  }

  function focusElement(element) {
    if (element && document.contains(element) && typeof element.focus === "function") {
      element.focus({ preventScroll: true });
    }
  }

  function focusCloseButton() {
    focusElement(document.getElementById(CLOSE_ID));
  }

  function restorePreviousFocus() {
    focusElement(previousFocus);
  }

  function syncFocusState() {
    var visible = modalIsVisible();
    if (visible && !wasVisible) {
      window.setTimeout(focusCloseButton, 0);
    }
    if (!visible && wasVisible) {
      window.setTimeout(restorePreviousFocus, 0);
    }
    wasVisible = visible;
  }

  document.addEventListener(
    "focusin",
    function (event) {
      if (!modalIsVisible() && event.target instanceof HTMLElement) {
        previousFocus = event.target;
      }
    },
    true
  );

  document.addEventListener(
    "keydown",
    function (event) {
      if (event.key !== "Escape" || !modalIsVisible()) {
        return;
      }
      var closeButton = document.getElementById(CLOSE_ID);
      if (!closeButton) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      closeButton.click();
    },
    true
  );

  var overlayObserver = new MutationObserver(syncFocusState);
  var bodyObserver = new MutationObserver(bindOverlay);

  function bindOverlay() {
    var overlay = overlayElement();
    if (!overlay || overlay === boundOverlay) {
      return;
    }
    if (boundOverlay) {
      overlayObserver.disconnect();
    }
    boundOverlay = overlay;
    overlayObserver.observe(overlay, {
      attributes: true,
      attributeFilter: ["style", "class", "hidden", "aria-hidden"],
    });
    syncFocusState();
  }

  function start() {
    bindOverlay();
    if (document.body) {
      bodyObserver.observe(document.body, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
