// Second Listen results page: the final system's overall metrics in plain words, from /api/v2/overview.
// Colours: our system = acid lime, where we started = electric cobalt (outline or dashed for the second baseline).
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const SETS = [
    { id: "pm_window_test", tab: "Conversations", unit: "window", what: "Doctor-patient conversations, cut into 30-second pieces: 47 consultations, about 70,000 words." },
    { id: "pm_turn_test", tab: "One speaker at a time", unit: "turn", what: "The same 47 consultations, one speaker's turn at a time." },
    { id: "eka_test", tab: "Short medical clips", unit: null, what: "Short medical clips from a different dataset (Eka): 30 speaker groups, about 16,000 words." },
  ];
  const st = { d: null, set: "pm_window_test", err: "critical" };
  const tip = $("#tip");
  const showTip = (e, html) => {
    tip.innerHTML = html; tip.hidden = false;
    tip.style.left = `${Math.min(e.pageX + 14, window.innerWidth - tip.offsetWidth - 8)}px`;
    tip.style.top = `${e.pageY + 14}px`;
  };
  const hideTip = () => { tip.hidden = true; };
  const r1 = (x) => (Math.round(x * 10) / 10).toString();

  function tiles(T) {
    return `<div class="tiles">
      <div class="tile"><b class="mono">${r1(T.word_accuracy)}%</b><span>of words written correctly</span></div>
      <div class="tile"><b class="mono">${r1(T.numbers.recall)}%</b><span>of numbers heard right<br>(doses, ages, dates)</span></div>
      <div class="tile"><b class="mono">${r1(T.negations.recall)}%</b><span>of "no" and "not" heard right</span></div>
      <div class="tile"><b class="mono">${r1(T.critical_error_rate)}%</b><span>of serious words still wrong<br>(numbers, doses, "not")</span></div>
    </div>`;
  }

  // before -> now, one row per measure (lower is better for all of them)
  function dumbbell(rows) {
    const W = 760, L = 230, R = 80, rowH = 46, H = rows.length * rowH + 30;
    const max = Math.max(...rows.flatMap((r) => [r.a, r.b])) * 1.08;
    const X = (v) => L + (W - L - R) * v / max;
    let g = `<g class="grid"><text x="${L}" y="14">0</text><text x="${W - R}" y="14" text-anchor="end">${r1(max)}</text></g>`;
    rows.forEach((r, i) => {
      const y = 34 + i * rowH, xa = X(r.a), xb = X(r.b);
      const better = Math.round(100 * (1 - r.b / r.a));
      g += `<g class="db" data-tip="${esc(`<b>${r.label}</b><br>Where we started: ${r1(r.a)}${r.unit}<br>Now: ${r1(r.b)}${r.unit}<br>${better}% lower`)}">
        <rect class="hit" x="0" y="${y - 18}" width="${W}" height="${rowH}"/>
        <text class="lab" x="${L - 14}" y="${y + 4}" text-anchor="end">${esc(r.label)}</text>
        <line class="track" x1="${Math.min(xa, xb)}" x2="${Math.max(xa, xb)}" y1="${y}" y2="${y}"/>
        <circle class="pa" cx="${xa}" cy="${y}" r="7"/><circle class="pb" cx="${xb}" cy="${y}" r="8"/>
        <text class="va mono" x="${xa + 12}" y="${y + 4}">${r1(r.a)}</text>
        <text class="vb mono" x="${xb - 12}" y="${y + 4}" text-anchor="end">${r1(r.b)}</text></g>`;
    });
    return `<svg viewBox="0 0 ${W} ${H}" class="dbchart" role="img" aria-label="Where we started against now">${g}</svg>
      <p class="keyline"><i class="kb"></i> now <i class="ka"></i> where we started (Whisper small) · lower is better</p>`;
  }

  // two horizontal bars per measure: our flags (lime) against Whisper's own confidence (cobalt outline)
  function pairBars(rows) {
    const W = 760, L = 230, R = 70, rowH = 64, H = rows.length * rowH;
    const X = (v) => (W - L - R) * v / 100;
    let g = "";
    rows.forEach((r, i) => {
      const y = i * rowH + 8;
      g += `<g class="pb2" data-tip="${esc(`<b>${r.label}</b><br>Our flags: ${r1(r.a)}%<br>Whisper's own confidence: ${r1(r.b)}%`)}">
        <rect class="hit" x="0" y="${y - 6}" width="${W}" height="${rowH}"/>
        <text class="lab" x="${L - 14}" y="${y + 22}" text-anchor="end">${esc(r.label)}</text>
        <path class="ours" d="M${L},${y} h${Math.max(2, X(r.a) - 4)} a4,4 0 0 1 4,4 v12 a4,4 0 0 1 -4,4 h-${Math.max(2, X(r.a) - 4)} z"/>
        <text class="v mono" x="${L + X(r.a) + 8}" y="${y + 15}">${r1(r.a)}%</text>
        <rect class="base" x="${L}" y="${y + 26}" width="${Math.max(2, X(r.b))}" height="18" rx="4"/>
        <text class="v dim mono" x="${L + X(r.b) + 8}" y="${y + 40}">${r1(r.b)}%</text></g>`;
    });
    return `<svg viewBox="0 0 ${W} ${H}" class="pbchart" role="img" aria-label="Our flags against Whisper's confidence">${g}</svg>
      <p class="keyline"><i class="kb"></i> our flags <i class="ko"></i> Whisper's own confidence · higher is better</p>`;
  }

  function listenChart(unit) {
    const L0 = st.d.listening[unit], E = st.err, xs = L0.x_seconds, maxX = 240, k = xs.findIndex((x) => x >= maxX) + 1;
    const series = [
      { label: "Where we started: listen to whole pieces", cls: "s-base", y: L0[E].start_whole },
      { label: "Where we started: with flags", cls: "s-small", y: L0[E].start_flags },
      { label: "Our system: with flags", cls: "s-ours", y: L0[E].final_flags },
    ];
    const W = 960, H = 360, l = 52, r = 250, t = 14, b = 40;
    const maxY = Math.max(...series.map((s) => s.y[0])) * 1.05;
    const X = (v) => l + (W - l - r) * v / maxX, Y = (v) => t + (H - t - b) * (1 - v / maxY);
    let g = `<g class="grid">`;
    const step = maxY > 100 ? 50 : maxY > 20 ? 10 : maxY > 5 ? 2 : 1;
    for (let v = 0; v <= maxY; v += step) g += `<line x1="${l}" x2="${W - r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${l - 8}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
    for (let v = 0; v <= maxX; v += 60) g += `<text x="${X(v)}" y="${H - b + 18}" text-anchor="middle">${v ? `${v / 60} min` : "0"}</text>`;
    g += `</g>`;
    let lines = "";
    const labs = series.map((s) => {
      lines += `<polyline class="ln ${s.cls}" points="${xs.slice(0, k).map((x, i) => `${X(x).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join(" ")}"/>`;
      return { y: Y(s.y[k - 1]), s };
    }).sort((a, c) => a.y - c.y);
    for (let i = 1; i < labs.length; i++) labs[i].y = Math.max(labs[i].y, labs[i - 1].y + 32);
    for (const { y, s } of labs) {
      const [a, c] = s.label.split(": ");
      lines += `<text class="dl ${s.cls}" x="${W - r + 10}" y="${y}">${esc(a)}<tspan x="${W - r + 10}" dy="15">${esc(c)}</tspan></text>`;
    }
    const i30 = xs.indexOf(30), [base, small, ours] = series;
    const what = E === "critical" ? "serious mistakes (numbers, doses, \"not\")" : "mistakes (serious ones count 10 times)";
    return `<div class="seg small" role="group" aria-label="Which mistakes">
        <button type="button" data-err="critical" aria-pressed="${E === "critical"}">Serious mistakes</button>
        <button type="button" data-err="weighted" aria-pressed="${E === "weighted"}">All mistakes</button></div>
      <svg viewBox="0 0 ${W} ${H}" class="lchart" role="img" aria-label="Mistakes left against minutes of listening"
        data-k="${k}" data-l="${l}" data-r="${r}" data-w="${W}">${g}${lines}<line class="xh" x1="0" x2="0" y1="${t}" y2="${H - b}" visibility="hidden"/></svg>
      <p class="rs-line">After <b>30 seconds</b> of listening, about <b>${r1(ours.y[i30])}</b> ${what} are left per
        consultation with our system, against <b>${r1(small.y[i30])}</b> where we started and <b>${r1(base.y[i30])}</b> when
        whole pieces are played. Before any listening there are ${r1(ours.y[0])} with our system and ${r1(small.y[0])} where we started.</p>`;
  }

  function expertTable() {
    const F = st.d.final, rows = [
      ["Word accuracy", (s) => `${F.transcripts[s].word_accuracy}%`, "share of the said words written correctly"],
      ["Word error rate (WER)", (s) => `${F.transcripts[s].wer}%`, "wrong, missing or extra words per 100 said"],
      ["Weighted WER", (s) => `${F.transcripts[s].weighted_wer}%`, "the same, with serious words counting 10x"],
      ["Critical error rate", (s) => `${F.transcripts[s].critical_error_rate}%`, "numbers, doses and negations that are wrong"],
      ["Numbers: precision / recall / F1", (s) => { const n = F.transcripts[s].numbers; return `${n.precision} / ${n.recall} / ${n.f1}`; }, "precision: numbers written that were said; recall: numbers said that were written"],
      ["Negations: precision / recall / F1", (s) => { const n = F.transcripts[s].negations; return `${n.precision} / ${n.recall} / ${n.f1}`; }, ""],
      ["Spotting wrong words: AUROC / AUPRC", (s) => `${F.detection[s].words.auroc} / ${F.detection[s].words.auprc}`, "AUROC: chance a wrong word ranks above a right one; AUPRC: precision across all flag levels"],
      ["Spotting dropped words: AUROC / AUPRC", (s) => `${F.detection[s].gaps.auroc} / ${F.detection[s].gaps.auprc}`, ""],
      ["Riskiest 10% flagged: precision / recall / F1", (s) => { const x = F.detection[s].words.top10; return `${x.precision} / ${x.recall} / ${x.f1}`; }, "precision: flags that are real mistakes; recall: mistakes that are flagged"],
      ["Riskiest 10% flagged: accuracy", (s) => `${F.detection[s].words.top10.accuracy}%`, "high mainly because most words are right"],
      ["Riskiest 10% flagged: serious mistakes caught", (s) => `${F.detection[s].words.top10.critical_recall}%`, ""],
    ];
    return `<table><thead><tr><th>Measure</th>${SETS.map((s) => `<th>${s.tab}</th>`).join("")}<th>What it means</th></tr></thead><tbody>${
      rows.map(([n, f, m]) => `<tr><th>${n}</th>${SETS.map((s) => `<td class="mono">${f(s.id)}</td>`).join("")}<td class="mean">${m}</td></tr>`).join("")
    }</tbody></table>`;
  }

  function render() {
    const d = st.d, s = SETS.find((x) => x.id === st.set);
    const T = d.final.transcripts[st.set], T0 = d.start.transcripts[st.set];
    const D = d.final.detection[st.set];
    $("#sets").innerHTML = SETS.map((x) => `<button type="button" data-set="${x.id}" aria-pressed="${x.id === st.set}">${x.tab}</button>`).join("");
    $("#what").textContent = s.what;
    const oneIn = Math.max(1, Math.round(100 / D.words.top10.precision));
    $("#page").innerHTML = `
      <section class="rs-card"><h3>1 · How accurate the transcript is</h3>${tiles(T)}
        <h4>Where we started → now</h4>
        ${dumbbell([
          { label: "Words wrong, per 100", a: T0.wer, b: T.wer, unit: "" },
          { label: "Mistakes, serious count 10x", a: T0.weighted_wer, b: T.weighted_wer, unit: "" },
          { label: "Serious words wrong, %", a: T0.critical_error_rate, b: T.critical_error_rate, unit: "%" },
          { label: "Words skipped, per 100", a: T0.deletions, b: T.deletions, unit: "" },
        ])}
        <p class="rs-line"><b>${Math.round(100 * (1 - T.wer / T0.wer))}% fewer mistakes</b> than where we started.
          Most of the gain is words the old model skipped: ${r1(T0.deletions)} per 100 then, ${r1(T.deletions)} now.</p>
        <p class="rs-caveat">Still to fix: ${r1(100 - T.numbers.precision)} in 100 numbers our model writes are wrong or were never said
          (the old model: ${r1(100 - T0.numbers.precision)}). That is why a person still checks the flagged words.</p>
      </section>

      <section class="rs-card"><h3>2 · How well we spot the mistakes</h3>
        <p class="rs-sub">Only ${r1(D.wrong_word_rate)}% of words are wrong, so finding them is the hard part. If a person checks just the
          <b>10% riskiest words</b> we flag:</p>
        <div class="tiles">
          <div class="tile"><b class="mono">${r1(D.words.top10.recall)}%</b><span>of all wrong words are among them</span></div>
          <div class="tile"><b class="mono">${r1(D.words.top10.critical_recall)}%</b><span>of the serious mistakes are among them</span></div>
          <div class="tile"><b class="mono">1 in ${oneIn}</b><span>flags is a real mistake<br>(random picking: 1 in ${Math.round(100 / D.wrong_word_rate)})</span></div>
          <div class="tile"><b class="mono">${D.words.auroc.toFixed(2)}</b><span>ranking score (1.0 = perfect, 0.5 = guessing)</span></div>
        </div>
        <h4>Our flags against Whisper's own confidence</h4>
        ${pairBars([
          { label: "Wrong words found", a: D.words.top10.recall, b: D.whisper_confidence.top10.recall },
          { label: "Serious mistakes found", a: D.words.top10.critical_recall, b: D.whisper_confidence.top10.critical_recall },
          { label: "Flags that are real mistakes", a: D.words.top10.precision, b: D.whisper_confidence.top10.precision },
        ])}
      </section>

      <section class="rs-card"><h3>3 · How much listening it saves</h3>
        ${s.unit ? `<p class="rs-sub">A person checks a consultation (about 9 minutes of audio) by listening. Lower line = fewer mistakes left
          after the same listening time.</p>${listenChart(s.unit)}`
          : `<p class="rs-sub">Not measured for these clips: each is only a few seconds long. Pick Conversations or One speaker.</p>`}
      </section>

      <details class="rs-table"><summary>All the numbers, with precision, recall, F1, AUROC</summary>${expertTable()}</details>`;
    hook();
  }

  function hook() {
    $$("#sets button").forEach((b) => b.addEventListener("click", () => { st.set = b.dataset.set; render(); }));
    $$("button[data-err]").forEach((b) => b.addEventListener("click", () => { st.err = b.dataset.err; render(); }));
    $$("[data-tip]").forEach((g) => {
      g.addEventListener("mousemove", (e) => showTip(e, g.dataset.tip));
      g.addEventListener("mouseleave", hideTip);
    });
    const svg = $(".lchart");
    if (!svg) return;
    const s = SETS.find((x) => x.id === st.set), L0 = st.d.listening[s.unit][st.err], xs = st.d.listening[s.unit].x_seconds;
    const k = +svg.dataset.k, l = +svg.dataset.l, r = +svg.dataset.r, W = +svg.dataset.w, xh = $(".xh", svg);
    svg.addEventListener("mousemove", (e) => {
      const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const i = Math.max(0, Math.min(k - 1, Math.round((p.x - l) / (W - l - r) * 240 / (xs[1] - xs[0]))));
      const x = l + (W - l - r) * xs[i] / 240;
      xh.setAttribute("x1", x); xh.setAttribute("x2", x); xh.setAttribute("visibility", "visible");
      showTip(e, `<b>${xs[i]} s of listening</b><br>Our system: ${r1(L0.final_flags[i])}<br>Where we started: ${r1(L0.start_flags[i])}` +
                 `<br>Whole pieces: ${r1(L0.start_whole[i])}<br><small>mistakes left per consultation</small>`);
    });
    svg.addEventListener("mouseleave", () => { xh.setAttribute("visibility", "hidden"); hideTip(); });
  }

  fetch("/api/v2/overview").then((r) => (r.ok ? r.json() : r.json().then((j) => Promise.reject(new Error(j.detail)))))
    .then((d) => { st.d = d; render(); })
    .catch((e) => { $("#page").innerHTML = `<p class="status">Could not load the results: ${esc(e.message)}</p>`; });
})();
