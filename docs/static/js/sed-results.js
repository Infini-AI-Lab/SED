/* Results cards: "View all numbers" expand/collapse + bar reveal on scroll. */
(function () {
  "use strict";
  var cards = document.querySelectorAll(".results-card");
  if (!cards.length) return;
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  Array.prototype.forEach.call(cards, function (card) {
    var full = card.querySelector(".results-card__full");
    var btn = card.querySelector(".results-card__toggle");
    if (!full || !btn) return;

    var open = false;
    var timer = 0;

    function setHeight(px) { full.style.height = px + "px"; }

    function expand() {
      open = true;
      card.classList.add("is-open");
      btn.setAttribute("aria-expanded", "true");
      full.removeAttribute("aria-hidden");
      if (reduce) { full.style.height = "auto"; return; }
      setHeight(0);
      // force layout so the transition starts from 0
      void full.offsetHeight;
      setHeight(full.scrollHeight);
      clearTimeout(timer);
      timer = setTimeout(function () { if (open) full.style.height = "auto"; }, 560);
    }

    function collapse() {
      open = false;
      btn.setAttribute("aria-expanded", "false");
      full.setAttribute("aria-hidden", "true");
      if (reduce) { card.classList.remove("is-open"); full.style.height = ""; return; }
      setHeight(full.scrollHeight);
      void full.offsetHeight;
      card.classList.remove("is-open");
      setHeight(0);
      clearTimeout(timer);
      timer = setTimeout(function () { if (!open) full.style.height = ""; }, 560);
      // keep the toggle in view when a tall table folds away
      var r = card.getBoundingClientRect();
      if (r.top < 0) card.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    }

    btn.addEventListener("click", function () { open ? collapse() : expand(); });
    full.setAttribute("aria-hidden", "true");

    // if the table is open, keep height in sync with content on resize
    window.addEventListener("resize", function () {
      if (open && full.style.height !== "auto") full.style.height = "auto";
    });
  });

  // bars grow in when the summary scrolls into view
  var strips = document.querySelectorAll(".bar-strip");
  if (!strips.length) return;
  if (!("IntersectionObserver" in window) || reduce) {
    Array.prototype.forEach.call(strips, function (s) { s.classList.add("is-inview"); });
    return;
  }
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { e.target.classList.add("is-inview"); io.unobserve(e.target); }
    });
  }, { threshold: 0.25 });
  Array.prototype.forEach.call(strips, function (s) { io.observe(s); });
})();
