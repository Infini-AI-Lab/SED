import { FiberNodes } from "https://cdn.jsdelivr.net/npm/recoat@0.1.0/dist/three.mjs";

var fiber = null;
var fiberEl = null;
var nodeEls = [];
var driverTimer = null;
var currentY = 0.12;
var currentX = 0.5;
var currentIndex = 0;

function waitForSize(el) {
  return new Promise(function (resolve) {
    function tick() {
      var r = el.getBoundingClientRect();
      if (r.width > 24 && r.height > 80) {
        resolve(r);
        return;
      }
      requestAnimationFrame(tick);
    }
    tick();
  });
}

function fracFromNode(index) {
  if (!fiberEl || !nodeEls.length || !nodeEls[index]) return { x: 0.5, y: 0.12 };
  var rect = fiberEl.getBoundingClientRect();
  var nodeRect = nodeEls[index].getBoundingClientRect();
  if (rect.height < 2) return { x: 0.5, y: 0.12 };
  return {
    x: 0.5,
    y: (nodeRect.top + nodeRect.height / 2 - rect.top) / rect.height,
  };
}

function dispatchFiberMouse(xFrac, yFrac) {
  if (!fiberEl) return;
  var rect = fiberEl.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  window.dispatchEvent(
    new MouseEvent("mousemove", {
      clientX: rect.left + rect.width * xFrac,
      clientY: rect.top + rect.height * yFrac,
      bubbles: true,
    })
  );
}

function startDriver() {
  stopDriver();
  driverTimer = setInterval(function () {
    dispatchFiberMouse(currentX, currentY);
  }, 40);
}

function stopDriver() {
  if (driverTimer) {
    clearInterval(driverTimer);
    driverTimer = null;
  }
}

function boot(el) {
  var rect = el.getBoundingClientRect();
  var w = Math.max(Math.round(rect.width), 1);
  var h = Math.max(Math.round(rect.height), 1);

  fiber = FiberNodes({
    container: el,
    width: w,
    height: h,
    nodeCount: 520,
    color: "#8fb8a8",
    highlightColor: "#5f9d84",
    hoverColor: "#176b52",
    bgColor: "#eef4f1",
    speed: 0.18,
    metallic: 0.08,
    nodeSize: 1.05,
    elongation: 0.62,
    mouseRadius: 3.0,
    mouseForce: 0.48,
    responsive: true,
  });
  fiber.start();
  fiber.resize(w, h);

  var t = fracFromNode(currentIndex);
  currentX = t.x;
  currentY = t.y;
  startDriver();
  dispatchFiberMouse(currentX, currentY);

  setTimeout(function () {
    var r = el.getBoundingClientRect();
    var rw = Math.max(Math.round(r.width), 1);
    var rh = Math.max(Math.round(r.height), 1);
    if (fiber && typeof fiber.resize === "function") fiber.resize(rw, rh);
    dispatchFiberMouse(currentX, currentY);
  }, 150);
}

window.__sedLoopFiber = {
  init: function (el, nodes) {
    if (!el || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    fiberEl = el;
    nodeEls = nodes || [];
    currentIndex = 0;

    waitForSize(el).then(function () {
      if (fiber) {
        fiber.destroy();
        fiber = null;
      }
      boot(el);
    });
  },

  sync: function (nodes) {
    nodeEls = nodes || nodeEls;
    if (fiber && fiberEl && typeof fiber.resize === "function") {
      var rs = fiberEl.getBoundingClientRect();
      fiber.resize(Math.max(Math.round(rs.width), 1), Math.max(Math.round(rs.height), 1));
    }
    var t = fracFromNode(currentIndex);
    currentY = t.y;
    dispatchFiberMouse(currentX, currentY);
  },

  resize: function (nodes) {
    if (nodes) nodeEls = nodes;
    if (fiber && fiberEl && typeof fiber.resize === "function") {
      var rr = fiberEl.getBoundingClientRect();
      fiber.resize(Math.max(Math.round(rr.width), 1), Math.max(Math.round(rr.height), 1));
    }
    var t = fracFromNode(currentIndex);
    currentX = t.x;
    currentY = t.y;
    dispatchFiberMouse(currentX, currentY);
  },

  goToStep: function (index, animate, nodes) {
    if (nodes) nodeEls = nodes;
    currentIndex = index;
    var target = fracFromNode(index);
    if (!animate || !window.anime) {
      currentX = target.x;
      currentY = target.y;
      dispatchFiberMouse(currentX, currentY);
      return;
    }
    var proxy = { y: currentY };
    window.anime({
      targets: proxy,
      y: target.y,
      duration: 850,
      easing: "easeInOutQuart",
      update: function () {
        currentY = proxy.y;
      },
    });
  },

  destroy: function () {
    stopDriver();
    if (fiber) {
      fiber.destroy();
      fiber = null;
    }
    fiberEl = null;
    nodeEls = [];
  },
};
