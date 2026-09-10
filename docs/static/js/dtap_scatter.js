(function () {
  "use strict";

  const BADGE_BASE = "static/images/badges/";
  const BADGE_VER = "dtapui2";
  const BADGE_RADIUS = 21;
  const HOVER_SCALE = 1.38;
  const LABEL_GAP = 4;

  const MODEL_META = {
    deepseek: { label: "DeepSeek V4 Flash", short: "DeepSeek", ring: "#1565C0" },
    glm: { label: "GLM-5.2", short: "GLM-5.2", ring: "#00897B" },
    kimi: { label: "Kimi K3", short: "Kimi K3", ring: "#C62828" },
  };

  const DEFENSE_SHORT = {
    no_defense: "No defense",
    llama_guard: "Llama Guard 3",
    safeharbor: "SafeHarbor",
    drift: "DRIFT",
    guardagent: "GuardAgent",
    sed: "SED",
  };

  let data = null;
  let chart = null;
  let activeModel = "all";
  let yMetric = "indirect";
  let badgeCache = {};
  let hoverIdx = -1;
  let layoutCache = null; // { truePx, repelled }

  function $(id) {
    return document.getElementById(id);
  }

  function defenseLegendBadge(defenseKey) {
    if (defenseKey === "no_defense") return null;
    return BADGE_BASE + "defense_" + defenseKey + ".png?v=" + BADGE_VER;
  }

  function filteredPoints() {
    if (!data) return [];
    return data.points.filter(function (p) {
      return activeModel === "all" || p.model === activeModel;
    });
  }

  function badgePath(model, defense) {
    return BADGE_BASE + model + "_" + defense + ".png?v=" + BADGE_VER;
  }

  function loadBadge(model, defense) {
    const src = badgePath(model, defense);
    if (badgeCache[src]) return badgeCache[src];
    badgeCache[src] = new Promise(function (resolve, reject) {
      const img = new Image();
      img.onload = function () {
        resolve(img);
      };
      img.onerror = reject;
      img.src = src;
    });
    return badgeCache[src];
  }

  function repelPixels(pixels, radius, bounds) {
    const disp = pixels.map(function (p) {
      return { x: p.x, y: p.y };
    });
    const minX = bounds.left + radius;
    const maxX = bounds.right - radius;
    const minY = bounds.top + radius;
    const maxY = bounds.bottom - radius - 28; // room for labels

    for (let iter = 0; iter < 400; iter++) {
      let moved = false;
      for (let i = 0; i < disp.length; i++) {
        for (let j = i + 1; j < disp.length; j++) {
          const dx = disp[j].x - disp[i].x;
          const dy = disp[j].y - disp[i].y;
          let dist = Math.hypot(dx, dy);
          if (dist < 2 * radius + 6) {
            if (dist < 1e-6) dist = 1;
            const push = (2 * radius + 6 - dist) / 2 + 0.5;
            const ux = dx / dist;
            const uy = dy / dist;
            disp[i].x -= ux * push;
            disp[i].y -= uy * push;
            disp[j].x += ux * push;
            disp[j].y += uy * push;
            moved = true;
          }
        }
      }
      for (let k = 0; k < disp.length; k++) {
        disp[k].x = Math.max(minX, Math.min(maxX, disp[k].x));
        disp[k].y = Math.max(minY, Math.min(maxY, disp[k].y));
      }
      if (!moved) break;
    }
    return disp;
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  /** Very soft pink → mint diagonal (DTap leaderboard style). */
  function drawTradeoffBackground(ctx, area) {
    const w = Math.max(1, Math.floor(area.right - area.left));
    const h = Math.max(1, Math.floor(area.bottom - area.top));
    const img = ctx.createImageData(w, h);
    const buf = img.data;
    // pale rose → near-white → pale mint
    const rose = [252, 228, 230];
    const mid = [248, 250, 249];
    const mint = [220, 242, 236];

    for (let j = 0; j < h; j++) {
      const yn = 1 - j / (h - 1 || 1);
      for (let i = 0; i < w; i++) {
        const xn = i / (w - 1 || 1);
        const t = Math.max(0, Math.min(1, (xn + (1 - yn)) / 2));
        let rgb;
        if (t < 0.5) {
          const u = t / 0.5;
          rgb = [
            lerp(rose[0], mid[0], u),
            lerp(rose[1], mid[1], u),
            lerp(rose[2], mid[2], u),
          ];
        } else {
          const u = (t - 0.5) / 0.5;
          rgb = [
            lerp(mid[0], mint[0], u),
            lerp(mid[1], mint[1], u),
            lerp(mid[2], mint[2], u),
          ];
        }
        const idx = (j * w + i) * 4;
        buf[idx] = rgb[0];
        buf[idx + 1] = rgb[1];
        buf[idx + 2] = rgb[2];
        buf[idx + 3] = 255;
      }
    }
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.drawImage(off, area.left, area.top);
  }

  function drawCornerLabels(ctx, area) {
    ctx.save();
    ctx.font =
      '600 10px "Noto Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textBaseline = "top";

    const bad = "⚠  Worst: vulnerable & weak";
    ctx.fillStyle = "#c45c6a";
    ctx.fillText(bad, area.left + 10, area.top + 10);

    const good = "★  Ideal: capable & safe";
    const gw = ctx.measureText(good).width;
    ctx.fillStyle = "#1a7a5c";
    ctx.fillText(good, area.right - gw - 10, area.bottom - 18);
    ctx.restore();
  }

  function drawBadgeAt(ctx, img, x, y, radius, opts) {
    const faded = opts.faded;
    const highlight = opts.highlight;
    const scale = highlight ? HOVER_SCALE : 1;
    const r = radius * scale;

    ctx.save();
    if (faded) ctx.globalAlpha = 0.32;

    ctx.beginPath();
    ctx.arc(x, y, r + 2, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.shadowColor = highlight ? "rgba(23,107,82,0.28)" : "rgba(0,0,0,0.12)";
    ctx.shadowBlur = highlight ? 16 : 5;
    ctx.shadowOffsetY = highlight ? 2 : 1;
    ctx.fill();
    ctx.shadowColor = "transparent";

    if (img && img.complete) {
      ctx.drawImage(img, x - r, y - r, r * 2, r * 2);
    }

    if (highlight) {
      ctx.beginPath();
      ctx.arc(x, y, r + 4, 0, Math.PI * 2);
      ctx.strokeStyle = "#176b52";
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawLabel(ctx, line1, line2, x, y, radius, highlight) {
    const scale = highlight ? HOVER_SCALE : 1;
    const r = radius * scale;
    const ly = y + r + LABEL_GAP + 2;
    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    if (highlight) {
      ctx.font =
        '600 11px "Noto Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      const tw = Math.max(
        ctx.measureText(line1).width,
        ctx.measureText(line2).width
      );
      const padX = 8;
      const padY = 4;
      const boxW = tw + padX * 2;
      const boxH = 30;
      ctx.fillStyle = "#171a18";
      roundRect(ctx, x - boxW / 2, ly - 2, boxW, boxH, 6);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.fillText(line1, x, ly);
      ctx.font =
        '500 10px "Noto Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillStyle = "rgba(255,255,255,0.78)";
      ctx.fillText(line2, x, ly + 13);
    } else {
      ctx.font =
        '600 10px "Noto Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillStyle = "#2a302e";
      ctx.fillText(line1, x, ly);
      ctx.font =
        '500 9px "Noto Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillStyle = "#6b736f";
      ctx.fillText(line2, x, ly + 12);
    }
    ctx.restore();
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawTooltip(ctx, pt, x, y, radius, metric) {
    const lines = [
      MODEL_META[pt.model].short + " · " + DEFENSE_SHORT[pt.defense],
      "Benign " + pt.benign.toFixed(1) + "%",
      (metric === "indirect" ? "Indirect" : "Direct") +
        " ASR " +
        pt[metric].toFixed(1) +
        "%",
    ];
    ctx.save();
    ctx.font =
      '500 11px "Noto Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    let maxW = 0;
    lines.forEach(function (t) {
      maxW = Math.max(maxW, ctx.measureText(t).width);
    });
    const pad = 10;
    const boxW = maxW + pad * 2;
    const boxH = 54;
    let bx = x - boxW / 2;
    let by = y + radius + 36;
    // keep on chart
    if (chart && chart.chartArea) {
      bx = Math.max(chart.chartArea.left + 4, Math.min(bx, chart.chartArea.right - boxW - 4));
      if (by + boxH > chart.chartArea.bottom - 4) {
        by = y - radius - boxH - 8;
      }
    }
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "rgba(0,0,0,0.08)";
    ctx.lineWidth = 1;
    ctx.shadowColor = "rgba(0,0,0,0.12)";
    ctx.shadowBlur = 8;
    roundRect(ctx, bx, by, boxW, boxH, 8);
    ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.stroke();

    ctx.fillStyle = "#171a18";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.font =
      '600 11px "Noto Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.fillText(lines[0], bx + pad, by + 8);
    ctx.font =
      '500 11px "Noto Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.fillStyle = "#5e6561";
    ctx.fillText(lines[1], bx + pad, by + 24);
    ctx.fillStyle = "#0d9488";
    ctx.fillText(lines[2], bx + pad, by + 38);
    ctx.restore();
  }

  const dtapBgPlugin = {
    id: "dtapBg",
    beforeDatasetsDraw: function (chartInst) {
      const area = chartInst.chartArea;
      if (!area) return;
      const ctx = chartInst.ctx;
      ctx.save();
      drawTradeoffBackground(ctx, area);
      drawCornerLabels(ctx, area);
      ctx.restore();
    },
  };

  const dtapBadgePlugin = {
    id: "dtapBadges",
    afterDatasetsDraw: function (chartInst) {
      const opts = chartInst.options.plugins.dtapBadges;
      if (!opts || !opts.badges || !opts.badges.length) return;

      const meta = chartInst.getDatasetMeta(0);
      if (!meta || !meta.data.length) return;

      const ctx = chartInst.ctx;
      const radius = opts.radius || BADGE_RADIUS;
      const pts = opts.pts;
      const metric = opts.metric;

      const truePx = meta.data.map(function (el) {
        return { x: el.x, y: el.y };
      });
      const repelled = repelPixels(truePx, radius + 2, chartInst.chartArea);
      layoutCache = { truePx: truePx, repelled: repelled };

      // connector lines for nudged badges (subtle)
      for (let i = 0; i < pts.length; i++) {
        const tp = truePx[i];
        const rp = repelled[i];
        if (Math.hypot(rp.x - tp.x, rp.y - tp.y) > 5) {
          ctx.save();
          ctx.globalAlpha = hoverIdx >= 0 && hoverIdx !== i ? 0.15 : 0.35;
          ctx.strokeStyle = "#9aa39e";
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(tp.x, tp.y);
          ctx.lineTo(rp.x, rp.y);
          ctx.stroke();
          ctx.beginPath();
          ctx.fillStyle = pts[i].modelRing || MODEL_META[pts[i].model].ring;
          ctx.arc(tp.x, tp.y, 2.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
      }

      // draw non-hovered first, then hovered on top (zoomed)
      const order = [];
      for (let i = 0; i < pts.length; i++) if (i !== hoverIdx) order.push(i);
      if (hoverIdx >= 0) order.push(hoverIdx);

      order.forEach(function (i) {
        const rp = repelled[i];
        const faded = hoverIdx >= 0 && hoverIdx !== i;
        const highlight = hoverIdx === i;
        drawBadgeAt(ctx, opts.badges[i], rp.x, rp.y, radius, {
          faded: faded,
          highlight: highlight,
        });
      });

      // Labels: SED first, then the undefended points, then the rest. A label is
      // skipped when it would overlap one already placed, and on narrow charts only
      // SED and undefended points are labelled (the rest show on hover/tap).
      const compact = chartInst.chartArea.right - chartInst.chartArea.left < 520;
      const prio = function (p) {
        return p.defense === "sed" ? 0 : p.defense === "no_defense" ? 1 : 2;
      };
      const labelOrder = [];
      for (let i = 0; i < pts.length; i++) if (i !== hoverIdx) labelOrder.push(i);
      labelOrder.sort(function (a, b) { return prio(pts[a]) - prio(pts[b]); });
      const placed = [];
      const overlaps = function (b) {
        return placed.some(function (o) {
          return b.x < o.x + o.w && b.x + b.w > o.x && b.y < o.y + o.h && b.y + b.h > o.y;
        });
      };
      ctx.save();
      ctx.font = '600 10px "Noto Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      labelOrder.forEach(function (i) {
        if (hoverIdx >= 0) return; // while hovering, only the hovered label is drawn
        const pt = pts[i];
        if (compact && prio(pt) > 1) return;
        const rp = repelled[i];
        const l1 = MODEL_META[pt.model].short;
        const l2 = DEFENSE_SHORT[pt.defense];
        const w = Math.max(ctx.measureText(l1).width, ctx.measureText(l2).width * 0.9) + 6;
        const box = { x: rp.x - w / 2, y: rp.y + radius + LABEL_GAP, w: w, h: 24 };
        if (overlaps(box)) return;
        placed.push(box);
        drawLabel(ctx, l1, l2, rp.x, rp.y, radius, false);
      });
      ctx.restore();
      if (hoverIdx >= 0) {
        const hp = pts[hoverIdx];
        const hr = repelled[hoverIdx];
        drawLabel(ctx, MODEL_META[hp.model].short, DEFENSE_SHORT[hp.defense], hr.x, hr.y, radius, true);
      }

      if (hoverIdx >= 0) {
        const rp = repelled[hoverIdx];
        const hoverR = radius * HOVER_SCALE;
        drawTooltip(ctx, pts[hoverIdx], rp.x, rp.y, hoverR, metric);
      }
    },
  };

  if (window.Chart && !window.__dtapBadgePluginRegistered) {
    Chart.register(dtapBgPlugin);
    Chart.register(dtapBadgePlugin);
    window.__dtapBadgePluginRegistered = true;
  }

  function hitTest(event, chartInst) {
    if (!layoutCache || !layoutCache.repelled) return -1;
    const rect = chartInst.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const r = BADGE_RADIUS + 10;
    // prefer topmost (higher index among overlaps) — scan reverse of draw order: champion/hover candidates
    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < layoutCache.repelled.length; i++) {
      const p = layoutCache.repelled[i];
      const d = Math.hypot(p.x - x, p.y - y);
      if (d <= r && d <= bestDist) {
        bestDist = d;
        best = i;
      }
    }
    return best;
  }

  let mouseLeaveBound = false;

  async function buildChart() {
    const canvas = $("dtap-scatter-canvas");
    if (!canvas || !window.Chart) return;

    const pts = filteredPoints();
    const badges = await Promise.all(
      pts.map(function (p) {
        return loadBadge(p.model, p.defense);
      })
    );

    hoverIdx = -1;
    layoutCache = null;
    if (chart) chart.destroy();

    chart = new Chart(canvas, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Defenses",
            data: pts.map(function (p) {
              return {
                x: p.benign,
                y: p[yMetric],
                defense: p.defenseLabel,
                model: p.modelLabel,
              };
            }),
            pointRadius: 0,
            pointHoverRadius: 0,
            hitRadius: 0,
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 550, easing: "easeOutQuart" },
        onHover: function (event, _elements, chartInst) {
          const native = event.native || event;
          const idx = hitTest(native, chartInst);
          if (idx !== hoverIdx) {
            hoverIdx = idx;
            chartInst.canvas.style.cursor = idx >= 0 ? "pointer" : "default";
            chartInst.draw();
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
          dtapBadges: {
            badges: badges,
            pts: pts,
            radius: BADGE_RADIUS,
            metric: yMetric,
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "Benign task success (%) — higher is better →",
              font: { size: 12, weight: "600" },
              color: "#5e6561",
            },
            min: 68,
            max: 100,
            grid: { color: "rgba(0,0,0,0.05)", borderDash: [3, 3] },
            ticks: { color: "#6b736f", font: { size: 11 } },
          },
          y: {
            title: {
              display: true,
              text: "Attack success rate (%) — lower is better ↑",
              font: { size: 12, weight: "600" },
              color: "#5e6561",
            },
            min: 0,
            suggestedMax: 100,
            grid: { color: "rgba(0,0,0,0.05)", borderDash: [3, 3] },
            ticks: { color: "#6b736f", font: { size: 11 } },
          },
        },
      },
    });

    if (!mouseLeaveBound) {
      mouseLeaveBound = true;
      canvas.addEventListener("mouseleave", function () {
        if (hoverIdx !== -1) {
          hoverIdx = -1;
          if (chart) chart.draw();
        }
      });
    }
  }

  function setActive(btn) {
    document.querySelectorAll(".dtap-ctrl[data-group]").forEach(function (el) {
      if (el.dataset.group === btn.dataset.group) {
        el.classList.toggle("active", el === btn);
      }
    });
    if (btn.dataset.group === "model") {
      activeModel = btn.dataset.value;
    } else if (btn.dataset.group === "metric") {
      yMetric = btn.dataset.value;
      const ylab = $("dtap-y-label");
      if (ylab) {
        ylab.textContent =
          yMetric === "indirect"
            ? "Macro-average indirect attack success (%) across CRM, workflow, and code"
            : "Macro-average direct attack success (%) across CRM, workflow, and code";
      }
    }
    buildChart();
  }

  function renderLegend() {
    const modelsEl = $("dtap-legend-models");
    const defsEl = $("dtap-legend-defenses");
    const frontierEl = $("dtap-legend-frontier");
    if (!data || !modelsEl || !defsEl) return;

    modelsEl.innerHTML = Object.keys(MODEL_META)
      .map(function (k) {
        const m = MODEL_META[k];
        const sample = BADGE_BASE + k + "_no_defense.png?v=" + BADGE_VER;
        return (
          '<span class="dtap-leg-item">' +
          '<img src="' +
          sample +
          '" alt="' + m.label + ' badge" class="dtap-leg-badge">' +
          m.label +
          "</span>"
        );
      })
      .join("");

    defsEl.innerHTML = data.defenses
      .map(function (d) {
        const bold = d.key === "sed" ? " dtap-leg-sed" : "";
        const logo = defenseLegendBadge(d.key);
        const icon = logo
          ? '<img src="' + logo + '" alt="' + d.label + ' chip" class="dtap-leg-badge">'
          : '<span class="dtap-leg-nodef">—</span>';
        return (
          '<span class="dtap-leg-item' +
          bold +
          '">' +
          icon +
          d.label +
          "</span>"
        );
      })
      .join("");

    if (frontierEl) {
      frontierEl.innerHTML =
        '<span class="dtap-leg-item dtap-leg-hint">↘ Ideal corner: high benign utility, low attack success — hover a badge to zoom in</span>';
    }
  }

  function initControls() {
    document.querySelectorAll(".dtap-ctrl").forEach(function (btn) {
      btn.addEventListener("click", function () {
        setActive(btn);
      });
    });
  }

  function init() {
    if (!$("dtap-scatter-root")) return;

    fetch("static/data/dtap_scatter.json")
      .then(function (r) {
        return r.json();
      })
      .then(function (json) {
        data = json;
        renderLegend();
        initControls();
        return buildChart();
      })
      .catch(function (err) {
        console.error("DTap scatter load failed:", err);
        const fb = $("dtap-scatter-fallback");
        if (fb) fb.hidden = false;
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
