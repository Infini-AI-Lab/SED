/* Figure 1: animated SED loop.
 * Loads the semantic SVG built by build_fig1_svg.py, swaps it in for the static PNG,
 * and drives a 7-step flow: packets travel along arrow centrelines while the node
 * that is "working" glows. Hover pauses; click a step chip or a labelled arrow to jump.
 * If anything fails (fetch, parse, no SVG support) the PNG simply stays in place. */
(function () {
  "use strict";

  var fig = document.querySelector(".overview-figure");
  if (!fig) return;
  var img = fig.querySelector("img");
  if (!img) return;

  // The PNG is only fetched when the vector version cannot be used.
  function usePng() {
    var src = img.getAttribute("data-src");
    if (src && img.getAttribute("src") !== src) img.setAttribute("src", src);
  }
  if (!window.fetch || !("getPointAtLength" in SVGPathElement.prototype)) { usePng(); return; }

  var SVG_URL = fig.getAttribute("data-svg") || "static/images/fig1/sed_loop.svg";
  var REDUCE = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var INK = "#171a18";
  var RED = "#e0433a";
  var GREEN = "#2f9e44";
  var PURPLE = "#7030a0";
  var BLUE = "#0070c0";

  var NODE_COLOR = {
    "node-input": "#5284c7",
    "node-library": "#946eb1",
    "node-tree": "#7030a0",
    "node-retrieval": "#946eb1",
    "node-agent": "#5284c7",
    "node-judge": "#f39a4a",
    "node-attribution": "#4ea72e",
    "node-memory": "#e0433a"
  };

  /* start/dur are fractions of the step duration */
  var STEPS = [
    {
      key: "query", num: 1, name: "Query", ms: 2600,
      desc: "A task arrives with untrusted content from the environment. The agent reads relevant policies from the library before it acts.",
      nodes: ["node-input", "node-library"],
      labels: ["arrow-query", "arrow-read"],
      banner: "banner-episode",
      flows: [
        { id: "flow-query", color: INK, start: 0.05, dur: 0.45 },
        { id: "flow-read", color: INK, start: 0.25, dur: 0.7 }
      ]
    },
    {
      key: "retrieve", num: 2, name: "Retrieve", ms: 2800,
      desc: "Stored policies are ranked by cosine similarity between each unit body and the query; the top-N are selected.",
      nodes: ["node-retrieval"],
      labels: ["arrow-to-agent"],
      banner: "banner-episode",
      flows: [
        { id: "flow-ray-a", color: PURPLE, start: 0.0, dur: 0.26, thin: true },
        { id: "flow-ray-b", color: BLUE, start: 0.13, dur: 0.26, thin: true },
        { id: "flow-ray-c", color: PURPLE, start: 0.26, dur: 0.26, thin: true },
        { id: "flow-ray-d", color: "#b79dd4", start: 0.39, dur: 0.26, thin: true },
        { id: "flow-to-agent", color: INK, start: 0.64, dur: 0.34 }
      ]
    },
    {
      key: "act", num: 3, name: "Act", ms: 2300,
      desc: "The frozen agent runs on the original query plus the top-N policies, using its tools and environment. Weights never change.",
      nodes: ["node-agent"],
      labels: ["arrow-trajectory"],
      banner: "banner-episode",
      flows: [
        { id: "flow-trajectory", color: INK, start: 0.4, dur: 0.55 }
      ]
    },
    {
      key: "judge", num: 4, name: "Judge", ms: 2700,
      desc: "After the episode, an external judge scores the full trajectory once. Benign or refused means no memory write; an attack success is diagnosed.",
      nodes: ["node-judge"],
      labels: [],
      banner: "banner-post",
      branch: true,
      flows: []
    },
    {
      key: "store", num: 5, name: "Store", ms: 2500,
      desc: "The judged harmful episode, with its diagnosis and evidence span, is written to episodic memory.",
      nodes: ["node-judge", "node-memory"],
      labels: ["arrow-store", "arrow-store-tag"],
      banner: "banner-post",
      flows: [
        { id: "flow-store", color: RED, start: 0.05, dur: 0.8 }
      ]
    },
    {
      key: "synth", num: 6, name: "Synthesize", ms: 2600,
      desc: "Harmful episodes are distilled into reusable policies and merged into the policy tree.",
      nodes: ["node-memory", "node-tree", "node-library"],
      labels: ["arrow-synth"],
      banner: "banner-post",
      flows: [
        { id: "flow-synth", color: RED, start: 0.05, dur: 0.75 }
      ]
    },
    {
      key: "update", num: 7, name: "Update", ms: 3000,
      desc: "Retrieved-policy statistics are updated, and the refreshed memory is read in the next episode. Every attack success becomes a defense.",
      nodes: ["node-attribution", "node-library"],
      labels: ["arrow-update"],
      banner: "banner-post",
      flows: [
        { id: "flow-green", color: GREEN, start: 0.0, dur: 0.42 },
        { id: "flow-update", color: INK, start: 0.38, dur: 0.6, dashed: true }
      ]
    }
  ];

  /* clicking a node jumps to the step where it is first active */
  var NODE_TO_STEP = {
    "node-input": 0, "node-library": 0, "node-retrieval": 1, "node-agent": 2,
    "node-judge": 3, "node-memory": 4, "node-tree": 5, "node-attribution": 6
  };
  var LABEL_TO_STEP = {
    "arrow-query": 0, "arrow-read": 0, "arrow-to-agent": 1, "arrow-trajectory": 2,
    "arrow-store": 4, "arrow-store-tag": 4, "arrow-synth": 5, "arrow-update": 6
  };

  fetch(SVG_URL, { credentials: "same-origin" })
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
    .then(mount)
    .catch(usePng);

  function mount(svgText) {
    var doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
    var svg = doc.documentElement;
    if (!svg || svg.nodeName.toLowerCase() !== "svg" || doc.querySelector("parsererror")) return;
    svg.removeAttribute("width");
    svg.removeAttribute("height");

    var stage = document.createElement("div");
    stage.className = "sed-fig1";
    stage.appendChild(document.importNode(svg, true));
    svg = stage.firstElementChild;

    var ui = buildControls();
    fig.insertBefore(stage, img);
    fig.insertBefore(ui.root, img.nextSibling);
    fig.classList.add("is-svg");

    var engine = new Engine(svg, ui, stage);
    engine.init();
  }

  function buildControls() {
    var root = document.createElement("div");
    root.className = "sed-fig1-ui";
    var chips = document.createElement("div");
    chips.className = "sed-fig1-steps";
    chips.setAttribute("role", "tablist");
    chips.setAttribute("aria-label", "SED loop steps");
    var buttons = STEPS.map(function (s, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "sed-fig1-step";
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", i === 0 ? "true" : "false");
      b.dataset.step = String(i);
      b.innerHTML = '<span class="sed-fig1-step__num">' + s.num + '</span>' +
        '<span class="sed-fig1-step__name">' + s.name + '</span>' +
        '<span class="sed-fig1-step__bar" aria-hidden="true"><i></i></span>';
      chips.appendChild(b);
      return b;
    });
    var caption = document.createElement("p");
    caption.className = "sed-fig1-caption";
    caption.setAttribute("aria-live", "polite");
    var hint = document.createElement("span");
    hint.className = "sed-fig1-hint";
    var touch = window.matchMedia("(hover: none)").matches;
    hint.textContent = REDUCE
      ? (touch ? "Tap a step to highlight it" : "Click a step to highlight it")
      : (touch ? "Auto-advances · swipe the diagram · tap a step to jump" : "Auto-advances · hover to pause · click a step or arrow to jump");
    root.appendChild(chips);
    root.appendChild(caption);
    root.appendChild(hint);
    return { root: root, buttons: buttons, caption: caption };
  }

  function Engine(svg, ui, stage) {
    this.svg = svg;
    this.ui = ui;
    this.stage = stage;
    this.step = 0;
    this.t0 = 0;
    this.paused = false;
    this.hover = false;
    this.visible = true;
    this.raf = 0;
    this.flowEls = {};
    this.packets = {};
    this.lenCache = {};
  }

  Engine.prototype.init = function () {
    var self = this;
    var svg = this.svg;

    // halo rects behind each node's outer box
    Object.keys(NODE_COLOR).forEach(function (id) {
      var g = svg.getElementById("f1-" + id);
      if (!g) return;
      var box = g.querySelector(".f1-box, .f1-tree-node");
      if (!box) return;
      var bb;
      try { bb = box.getBBox(); } catch (e) { return; }
      if (id === "node-tree") { try { bb = g.getBBox(); } catch (e) { return; } }
      var pad = id === "node-tree" ? 6 : 4;
      var halo = document.createElementNS(svg.namespaceURI, "rect");
      halo.setAttribute("class", "f1-halo");
      halo.setAttribute("x", bb.x - pad);
      halo.setAttribute("y", bb.y - pad);
      halo.setAttribute("width", bb.width + pad * 2);
      halo.setAttribute("height", bb.height + pad * 2);
      halo.setAttribute("rx", (parseFloat(box.getAttribute("rx")) || 10) + pad);
      halo.setAttribute("stroke", NODE_COLOR[id]);
      halo.style.color = NODE_COLOR[id];
      g.insertBefore(halo, g.firstChild);
      g.classList.add("f1-clickable");
      g.addEventListener("click", function () { self.jump(NODE_TO_STEP[id], true); });
    });

    Object.keys(LABEL_TO_STEP).forEach(function (id) {
      var g = svg.getElementById("f1-" + id);
      if (!g) return;
      g.classList.add("f1-clickable");
      g.addEventListener("click", function () { self.jump(LABEL_TO_STEP[id], true); });
    });

    // flow paths + packets
    var packetLayer = svg.getElementById("f1-packets");
    Array.prototype.forEach.call(svg.querySelectorAll(".f1-flow"), function (p) {
      var id = p.getAttribute("data-flow");
      self.flowEls[id] = p;
      var len = p.getTotalLength();
      self.lenCache[id] = len;
      p.style.strokeDasharray = len + " " + len;
      p.style.strokeDashoffset = String(len);
      p.style.opacity = "0";

      var halo = document.createElementNS(svg.namespaceURI, "circle");
      halo.setAttribute("class", "f1-packet-halo");
      halo.setAttribute("r", "9");
      var dot = document.createElementNS(svg.namespaceURI, "circle");
      dot.setAttribute("class", "f1-packet");
      dot.setAttribute("r", "4.2");
      halo.style.opacity = dot.style.opacity = "0";
      packetLayer.appendChild(halo);
      packetLayer.appendChild(dot);
      self.packets[id] = { halo: halo, dot: dot };
    });

    // controls
    this.ui.buttons.forEach(function (b, i) {
      b.addEventListener("click", function () { self.jump(i, true); });
    });
    this.stage.addEventListener("mouseenter", function () { self.hover = true; });
    this.stage.addEventListener("mouseleave", function () { self.hover = false; self.t0 = performance.now() - self.elapsed; });
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) self.visible = false; else { self.visible = true; self.t0 = performance.now() - self.elapsed; }
    });
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          self.inView = e.isIntersecting;
          if (e.isIntersecting) self.t0 = performance.now() - self.elapsed;
        });
      }, { threshold: 0.15 });
      io.observe(this.stage);
    } else {
      this.inView = true;
    }

    this.elapsed = 0;
    this.applyStep(0);
    if (REDUCE) {
      this.renderStatic(0);
      return;
    }
    this.t0 = performance.now();
    this.loop();
  };

  Engine.prototype.jump = function (i, manual) {
    if (i == null || i < 0) return;
    this.step = i % STEPS.length;
    this.elapsed = 0;
    this.t0 = performance.now();
    this.applyStep(this.step);
    if (REDUCE) this.renderStatic(this.step);
    if (manual) {
      // brief hold so the click feels deliberate
      this.holdUntil = performance.now() + 600;
    }
  };

  // On narrow screens the diagram scrolls horizontally; keep the active nodes in view.
  Engine.prototype.followActive = function (s) {
    var stage = this.stage;
    if (stage.scrollWidth <= stage.clientWidth + 4 || !s.nodes.length) return;
    var svg = this.svg;
    var vb = svg.viewBox.baseVal;
    var scale = svg.getBoundingClientRect().width / vb.width;
    var minX = Infinity, maxX = -Infinity;
    s.nodes.forEach(function (n) {
      var g = svg.getElementById("f1-" + n);
      if (!g) return;
      var bb;
      try { bb = g.getBBox(); } catch (e) { return; }
      minX = Math.min(minX, bb.x);
      maxX = Math.max(maxX, bb.x + bb.width);
    });
    if (!isFinite(minX)) return;
    var center = ((minX + maxX) / 2) * scale;
    var target = Math.max(0, Math.min(center - stage.clientWidth / 2, stage.scrollWidth - stage.clientWidth));
    if (stage.scrollTo) stage.scrollTo({ left: target, behavior: REDUCE ? "auto" : "smooth" });
    else stage.scrollLeft = target;
  };

  Engine.prototype.applyStep = function (i) {
    var s = STEPS[i];
    var svg = this.svg;
    var active = {};
    s.nodes.forEach(function (n) { active[n] = true; });
    Object.keys(NODE_COLOR).forEach(function (id) {
      var g = svg.getElementById("f1-" + id);
      if (g) g.classList.toggle("is-active", !!active[id]);
    });
    var lit = {};
    s.labels.forEach(function (n) { lit[n] = true; });
    Object.keys(LABEL_TO_STEP).forEach(function (id) {
      var g = svg.getElementById("f1-" + id);
      if (g) g.classList.toggle("is-lit", !!lit[id]);
    });
    var green = svg.getElementById("f1-arrow-green");
    if (green) green.classList.toggle("is-lit", s.key === "update");
    ["banner-episode", "banner-post"].forEach(function (id) {
      var g = svg.getElementById("f1-" + id);
      if (g) g.classList.toggle("is-lit", s.banner === id);
    });
    var judge = svg.getElementById("f1-node-judge");
    if (judge) judge.classList.toggle("is-branch", !!s.branch);
    this.stage.setAttribute("data-step", s.key);
    this.stage.classList.add("is-live");
    this.followActive(s);

    this.ui.buttons.forEach(function (b, k) {
      b.classList.toggle("is-active", k === i);
      b.setAttribute("aria-selected", k === i ? "true" : "false");
      if (k === i && b.scrollIntoView && b.parentNode.scrollWidth > b.parentNode.clientWidth) {
        b.scrollIntoView({ block: "nearest", inline: "center", behavior: REDUCE ? "auto" : "smooth" });
      }
    });
    this.ui.caption.innerHTML = '<b>' + s.num + ' · ' + s.name + '</b> ' + s.desc;

    // reset flows not in this step
    var inStep = {};
    s.flows.forEach(function (f) { inStep[f.id] = true; });
    var self = this;
    Object.keys(this.flowEls).forEach(function (id) {
      if (!inStep[id]) self.setFlow(id, 0, null);
    });
  };

  Engine.prototype.setFlow = function (id, u, flow) {
    var p = this.flowEls[id];
    var len = this.lenCache[id];
    var pk = this.packets[id];
    if (!p || !pk) return;
    if (u <= 0) {
      p.style.opacity = "0";
      p.style.strokeDashoffset = String(len);
      pk.halo.style.opacity = pk.dot.style.opacity = "0";
      return;
    }
    p.style.opacity = u >= 1 ? "0.55" : "0.95";
    p.style.stroke = flow.color;
    p.style.strokeWidth = flow.thin ? "2.2" : "3.2";
    p.style.strokeDasharray = len + " " + len;
    p.style.strokeDashoffset = String(len * (1 - u));
    var head = u >= 1 ? len : len * u;
    var pt = p.getPointAtLength(head);
    pk.halo.setAttribute("cx", pt.x); pk.halo.setAttribute("cy", pt.y);
    pk.dot.setAttribute("cx", pt.x); pk.dot.setAttribute("cy", pt.y);
    pk.dot.style.fill = flow.color;
    pk.halo.style.fill = flow.color;
    var vis = u >= 1 ? 0 : Math.min(1, u * 6) * Math.min(1, (1 - u) * 6);
    pk.dot.style.opacity = String(vis);
    pk.halo.style.opacity = String(vis * 0.45);
  };

  Engine.prototype.renderStatic = function (i) {
    var self = this;
    STEPS[i].flows.forEach(function (f) { self.setFlow(f.id, 1, f); });
  };

  Engine.prototype.loop = function () {
    var self = this;
    this.raf = requestAnimationFrame(function (now) {
      var running = !self.hover && self.visible && self.inView !== false && !(self.holdUntil && now < self.holdUntil);
      if (running) {
        if (self.holdUntil) { self.holdUntil = 0; self.t0 = now - self.elapsed; }
        self.elapsed = now - self.t0;
      } else {
        self.t0 = now - self.elapsed;
      }
      var s = STEPS[self.step];
      var t = Math.min(1, self.elapsed / s.ms);
      var bar = self.ui.buttons[self.step].querySelector(".sed-fig1-step__bar i");
      if (bar) bar.style.transform = "scaleX(" + t + ")";
      s.flows.forEach(function (f) {
        var u = (t - f.start) / f.dur;
        u = u < 0 ? 0 : u > 1 ? 1 : easeInOut(u);
        self.setFlow(f.id, u, f);
      });
      if (running && self.elapsed >= s.ms) {
        var prev = self.ui.buttons[self.step].querySelector(".sed-fig1-step__bar i");
        if (prev) prev.style.transform = "scaleX(0)";
        self.jump(self.step + 1, false);
      }
      self.loop();
    });
  };

  function easeInOut(u) {
    return u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
  }
})();
