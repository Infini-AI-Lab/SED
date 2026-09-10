/**
 * SED vertical loop — row-synced spine + subtle FiberNodes.
 */
(function () {
  var STEPS = ["retrieve", "act", "judge", "learn"];
  var STEP_MS = 4200;

  var root = document.getElementById("sed-loop");
  if (!root) return;

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var active = 0;
  var timer = null;
  var paused = false;

  var inner = root.querySelector(".sed-loop__inner");
  var fiberEl = document.getElementById("sed-loop-fiber");
  var spine = root.querySelector(".sed-loop__spine");
  var rows = Array.prototype.slice.call(root.querySelectorAll(".sed-loop__row"));
  var nodes = Array.prototype.slice.call(root.querySelectorAll(".sed-loop__node"));
  var cards = Array.prototype.slice.call(root.querySelectorAll(".sed-loop__card"));

  var progressFill = spine && spine.querySelector(".sed-loop__progress-fill");
  var flowPath = spine && spine.querySelector(".sed-loop__flow");
  var packet = spine && spine.querySelector(".sed-loop__packet");
  var trackLen = 0;
  var flowLen = 0;
  var flowAnim = null;
  var spinePoints = [];

  function isMobile() {
    return window.matchMedia("(max-width: 720px)").matches;
  }

  function nodeCenters() {
    if (!inner) return [];
    var innerRect = inner.getBoundingClientRect();
    return nodes.map(function (node) {
      var r = node.getBoundingClientRect();
      return {
        x: r.left + r.width / 2 - innerRect.left,
        y: r.top + r.height / 2 - innerRect.top,
      };
    });
  }

  function layoutSpine() {
    if (!spine || !inner || isMobile()) return;
    var innerRect = inner.getBoundingClientRect();
    var h = Math.max(innerRect.height, 1);
    var w = parseFloat(getComputedStyle(root).getPropertyValue("--sed-rail")) || 84;

    spine.setAttribute("viewBox", "0 0 " + w + " " + h);
    spine.style.height = h + "px";

    spinePoints = nodeCenters();
    if (spinePoints.length < 2) return;

    var pathD =
      "M " +
      spinePoints[0].x +
      " " +
      spinePoints[0].y +
      spinePoints
        .slice(1)
        .map(function (p) {
          return " L " + p.x + " " + p.y;
        })
        .join("");

    if (progressFill) progressFill.setAttribute("d", pathD);
    if (flowPath) flowPath.setAttribute("d", pathD);

    trackLen = progressFill && progressFill.getTotalLength ? progressFill.getTotalLength() : 0;
    flowLen = flowPath && flowPath.getTotalLength ? flowPath.getTotalLength() : 0;

    if (progressFill && trackLen) {
      progressFill.style.strokeDasharray = String(trackLen);
      if (!progressFill.style.strokeDashoffset) {
        progressFill.style.strokeDashoffset = String(trackLen);
      }
    }
    if (flowPath && flowLen) {
      flowPath.style.strokeDasharray = "16 " + (flowLen - 16);
    }
  }

  function sizeFiberBox() {
    if (!fiberEl || !root) return;
    var rail = parseFloat(getComputedStyle(root).getPropertyValue("--sed-rail")) || 84;
    fiberEl.style.width = rail + 36 + "px";
    fiberEl.style.height = root.offsetHeight + "px";
  }

  function initFiber() {
    if (!fiberEl || reduced) return;
    function tryInit() {
      if (!window.__sedLoopFiber) {
        setTimeout(tryInit, 50);
        return;
      }
      layoutSpine();
      sizeFiberBox();
      window.__sedLoopFiber.init(fiberEl, nodes);
    }
    tryInit();
  }

  function pulseFiber(index, animate) {
    if (!window.__sedLoopFiber || reduced) return;
    window.__sedLoopFiber.goToStep(index, animate, nodes);
  }

  function animateFlow(direction) {
    if (!flowPath || !flowLen || !window.anime || reduced) return;
    if (flowAnim) flowAnim.pause();
    flowAnim = window.anime({
      targets: flowPath,
      strokeDashoffset: direction === "up" ? [0, -flowLen] : [0, flowLen],
      duration: direction === "up" ? 1100 : 2200,
      easing: "linear",
      loop: direction !== "up",
    });
  }

  function setActive(index, animate) {
    active = index;
    root.dataset.step = STEPS[index];

    nodes.forEach(function (node, i) {
      node.classList.toggle("is-active", i === index);
      node.classList.toggle("is-done", i < index);
    });

    cards.forEach(function (card, i) {
      card.classList.toggle("is-active", i === index);
    });

    layoutSpine();
    pulseFiber(index, animate);

    if (progressFill && trackLen) {
      var pct = index / (STEPS.length - 1);
      var offset = trackLen * (1 - pct);
      if (animate && window.anime && !reduced) {
        window.anime({
          targets: progressFill,
          strokeDashoffset: offset,
          duration: 750,
          easing: "easeInOutQuart",
        });
      } else {
        progressFill.style.strokeDashoffset = String(offset);
      }
    }

    var pos = spinePoints[index];
    if (packet && pos) {
      if (window.anime && animate && !reduced) {
        window.anime({
          targets: packet,
          cx: pos.x,
          cy: pos.y,
          duration: 780,
          easing: "easeInOutQuart",
        });
      } else {
        packet.setAttribute("cx", String(pos.x));
        packet.setAttribute("cy", String(pos.y));
      }
    }

    if (STEPS[index] === "learn" && animate && !reduced) {
      animateFlow("up");
      setTimeout(function () {
        animateFlow("down");
      }, 1150);
    } else if (animate && !reduced) {
      animateFlow("down");
    }
  }

  function nextStep() {
    setActive((active + 1) % STEPS.length, true);
  }

  function startTimer() {
    if (reduced || paused) return;
    clearInterval(timer);
    timer = setInterval(nextStep, STEP_MS);
  }

  function relayout() {
    layoutSpine();
    sizeFiberBox();
    if (window.__sedLoopFiber) {
      window.__sedLoopFiber.resize(nodes);
    }
    setActive(active, false);
  }

  function bindInteractions() {
    root.addEventListener("click", function (e) {
      var node = e.target.closest(".sed-loop__node");
      var card = e.target.closest(".sed-loop__card");
      var idx = -1;
      if (node) idx = nodes.indexOf(node);
      if (card && idx < 0) idx = cards.indexOf(card);
      if (idx >= 0) {
        setActive(idx, true);
        startTimer();
      }
    });

    root.addEventListener("keydown", function (e) {
      if (e.key !== "Enter" && e.key !== " ") return;
      var node = e.target.closest(".sed-loop__node");
      if (!node) return;
      e.preventDefault();
      var idx = nodes.indexOf(node);
      if (idx >= 0) {
        setActive(idx, true);
        startTimer();
      }
    });

    root.addEventListener("mouseenter", function () {
      paused = true;
      clearInterval(timer);
    });
    root.addEventListener("mouseleave", function () {
      paused = false;
      startTimer();
    });
    root.addEventListener("focusin", function () {
      paused = true;
      clearInterval(timer);
    });
    root.addEventListener("focusout", function (e) {
      if (!root.contains(e.relatedTarget)) {
        paused = false;
        startTimer();
      }
    });

    var resizeTimer;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(relayout, 120);
    });

    if (window.ResizeObserver && root) {
      new ResizeObserver(function () {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(relayout, 80);
      }).observe(root);
    }
  }

  function initReduced() {
    root.classList.add("is-reduced");
    cards.forEach(function (c) {
      c.setAttribute("aria-hidden", "false");
    });
    setActive(0, false);
  }

  function initAnimated() {
    layoutSpine();
    if (progressFill && trackLen) {
      progressFill.style.strokeDashoffset = String(trackLen);
    }
    setActive(0, false);
    animateFlow("down");
    startTimer();

    function startFiber() {
      layoutSpine();
      sizeFiberBox();
      initFiber();
    }

    if (document.readyState === "complete") {
      startFiber();
    } else {
      window.addEventListener("load", startFiber, { once: true });
    }

    requestAnimationFrame(function () {
      relayout();
    });
    setTimeout(function () {
      relayout();
      if (window.__sedLoopFiber) {
        window.__sedLoopFiber.resize(nodes);
      }
    }, 400);
  }

  bindInteractions();
  if (reduced) initReduced();
  else initAnimated();
})();
