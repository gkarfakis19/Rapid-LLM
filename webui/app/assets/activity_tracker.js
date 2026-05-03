(function () {
  if (window.rapidWebuiActivityTrackerInstalled) {
    return;
  }

  window.rapidWebuiActivityTrackerInstalled = true;
  window.rapidWebuiLastActivity = Date.now();

  const markActive = function () {
    window.rapidWebuiLastActivity = Date.now();
  };

  ["click", "keydown", "pointerdown", "pointermove", "scroll", "touchstart", "wheel"].forEach(function (eventName) {
    window.addEventListener(eventName, markActive, { capture: true, passive: true });
  });

  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) {
      markActive();
    }
  });
})();
