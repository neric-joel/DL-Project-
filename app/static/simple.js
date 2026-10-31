// Second Listen, simple view: Whisper (left) and our model (right) side by side, filling in as the audio plays.
// Our model = fine-tuned large-v3-turbo + verifier (saved results); Whisper = whisper-small, live or saved.
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const enc = encodeURIComponent;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (s) => { s = Math.max(0, Math.floor(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
  const OURS = "turbo-lora-mixed", WHISPER = "small";
  const TH = 0.25;    // verifier score (expected weighted error) at which a word or gap is flagged
  const PAD = 0.6;    // seconds of context around a flagged clip

  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) {
      let msg = `${r.status} ${r.statusText}`;
      try { const j = await r.json(); if (j.detail) msg = j.detail; } catch (_) { /* not JSON */ }
      throw new Error(msg);
    }
    return r.json();
  }

  const audio = new Audio();
  audio.preload = "auto";
  const rowsEl = $("#rows"), playBtn = $("#play"), seek = $("#seek");
  let C = null;            // {id, rows: [...], byId: Map}
  let items = [];          // revealable elements: {el, t}
  let dirty = true, lastNow = -1, stopAt = null;
  let src = "live", stream = null, streamStarted = false, decodedUntil = 0;

  const status = (html) => { $("#status").innerHTML = html || ""; };
  // fillers (um, uh, ...) are left out of the comparison: the human transcripts keep them, Whisper-small drops them
  const FILLER = /^(um+|uh+|erm*|hmm*|mm+|ah+|oh)$/;
  const norm = (t) => { const x = String(t).toLowerCase().replace(/[^a-z0-9']/g, ""); return FILLER.test(x) ? "" : x; };

  // longest common subsequence of normalised words: which words each side shares with the other
  function diff(a, b) {
    const A = a.map((w) => norm(w.text)), B = b.map((w) => norm(w.text));
    const n = A.length, m = B.length, L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
      L[i][j] = A[i] && A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    const ka = new Uint8Array(n), kb = new Uint8Array(m);
    let i = 0, j = 0;
    while (i < n && j < m) {
      if (A[i] && A[i] === B[j]) { ka[i++] = 1; kb[j++] = 1; }
      else if (L[i + 1][j] >= L[i][j + 1]) i++;
      else j++;
    }
    A.forEach((x, k) => { if (!x) ka[k] = 1; });   // punctuation-only tokens never count as changes
    B.forEach((x, k) => { if (!x) kb[k] = 1; });
    return [ka, kb];
  }

  // ---- building the two columns ------------------------------------------------------------
  function cellWords(cell, row, words, gaps, kind, keep) {
    const gapAfter = new Map();
    for (const g of gaps || []) if ((g.score ?? 0) >= TH) gapAfter.set(g.after, g);
    let html = `<span class="who">${esc(row.role)} · ${fmt(row.start)}</span>`;
    const gapFix = kind === "ours" ? row.gapFix : new Map();
    const gapHTML = (g, k) => gapFix.has(k)
      ? `<span class="w fixed" data-t="${g.t ?? row.start}">${esc(gapFix.get(k))}</span> `
      : g.ok ? "" : `<span class="gap" data-t="${g.t ?? row.start}" title="words may be missing${g.suggestion ? `: ${esc(g.suggestion)}` : ""}">‸</span>`;
    if (gapAfter.has(-1)) html += gapHTML(gapAfter.get(-1), -1);
    words.forEach((w, k) => {
      if (kind === "ours" && w.fixed !== undefined) {   // corrected in Review ("" = folded into the first word of the fix)
        if (w.fixed) html += `<span class="w fixed" data-t="${w.t0}" data-t1="${w.t1}">${esc(w.fixed)}</span> `;
      } else {
        const cls = ["w"];
        if (!keep[k]) cls.push(kind === "ours" ? "chg" : "cut");
        if (kind === "ours" && (w.score ?? 0) >= TH && !w.ok) cls.push("flag");
        html += `<span class="${cls.join(" ")}" data-t="${w.t0}" data-t1="${w.t1}">${esc(w.text)}</span> `;
      }
      if (gapAfter.has(k)) html += gapHTML(gapAfter.get(k), k);
    });
    cell.innerHTML = html;
  }

  function fillRow(row) {
    const wh = row.whisper, ours = row.ours;
    const [ka, kb] = wh ? diff(wh.words, ours.words) : [null, new Uint8Array(ours.words.length).fill(1)];
    cellWords(row.oEl, row, ours.words, ours.gaps, "ours", kb);
    if (wh) cellWords(row.wEl, row, wh.words, [], "whisper", ka);
    else row.wEl.innerHTML = `<span class="who">${esc(row.role)} · ${fmt(row.start)}</span>` +
      `<span class="pending">${src === "live" ? "Whisper is transcribing…" : "no saved transcript"}</span>`;
    dirty = true;
  }

  function build(oursPayload, whisperPayload) {
    const saved = new Map((whisperPayload?.units || []).map((u) => [u.id, u]));
    rowsEl.innerHTML = "";
    const rows = [...oursPayload.units].sort((a, b) => a.start - b.start).map((u) => {
      const el = document.createElement("div");
      el.className = "row";
      el.innerHTML = `<div class="cell whisper"></div><div class="cell ours"></div>`;
      rowsEl.appendChild(el);
      return { id: u.id, role: u.role || "speaker", start: u.start, end: u.end, ours: u, saved: saved.get(u.id) || null,
               gapFix: new Map(),
               whisper: src === "saved" ? saved.get(u.id) || null : null, el, wEl: el.children[0], oEl: el.children[1] };
    });
    C.rows = rows; C.byId = new Map(rows.map((r) => [r.id, r]));
    rows.forEach(fillRow);
    stats();
    buildSpots();
  }

  function stats() {
    const wrong = (us) => us.reduce((n, u) => n + u.words.filter((w) => w.wrong === 1).length, 0);
    const known = C.rows.some((r) => r.ours.words.some((w) => w.wrong !== null && w.wrong !== undefined));
    const total = C.rows.reduce((n, r) => n + r.ours.words.length, 0);
    const o = wrong(C.rows.map((r) => r.ours)), w = wrong(C.rows.filter((r) => r.saved).map((r) => r.saved));
    $("#stat-o").innerHTML = known ? `${o} words wrong<br>of ${total}` : "";
    $("#stat-w").innerHTML = known ? `${w} words wrong<br>${src === "live" ? "(saved run)" : ""}` : "";
  }

  // ---- revealing words as the audio plays --------------------------------------------------
  function collect() {
    items = $$(".w, .gap", rowsEl).map((el) => ({ el, t: +el.dataset.t })).sort((a, b) => a.t - b.t);
    dirty = false; lastNow = -1;
  }
  function tick() {
    const now = audio.currentTime || 0;
    if (dirty) collect();
    if (now !== lastNow) {
      for (const it of items) it.el.classList.toggle("on", it.t <= now + 0.05);
      let newest = null;
      for (const r of C?.rows || []) {
        const on = r.start <= now + 0.05;
        if (on && !r.el.classList.contains("on")) newest = r;
        r.el.classList.toggle("on", on);
        r.el.classList.toggle("now", on && now <= r.end + 0.5);
      }
      if (newest && !audio.paused) newest.oEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
      lastNow = now;
    }
    if (stopAt !== null && now >= stopAt) { audio.pause(); stopAt = null; }
    $("#time").textContent = `${fmt(now)} / ${fmt(audio.duration || C?.duration || 0)}`;
    if (!seek.matches(":active")) seek.value = String(now);
    if (src === "live" && streamStarted) {
      const behind = decodedUntil < now - 1 && !stream?.done;
      status(stream?.done ? "Whisper finished transcribing this consultation live."
        : `Whisper is transcribing live: ${fmt(decodedUntil)} of audio done${behind ? " (it is behind the audio)" : ""}.`);
    }
    requestAnimationFrame(tick);
  }

  // ---- live Whisper ----------------------------------------------------------------------------
  function closeStream() { if (stream) { stream.close(); stream = null; } streamStarted = false; decodedUntil = 0; }
  function startStream() {
    closeStream();
    streamStarted = true;
    status("Starting Whisper…");
    const es = new EventSource(`/api/v2/live/${enc(C.id)}`);
    stream = es;
    es.addEventListener("turn", (e) => {
      const u = JSON.parse(e.data), row = C.byId.get(u.id);
      if (row) { row.whisper = u; fillRow(row); }
    });
    es.addEventListener("progress", (e) => { decodedUntil = JSON.parse(e.data).decoded_until || decodedUntil; });
    es.addEventListener("done", () => { es.done = true; es.close(); });
    es.addEventListener("error", (e) => {
      let msg = "the live stream stopped";
      try { msg = JSON.parse(e.data).message; } catch (_) { /* connection error, no payload */ }
      if (es.done) return;
      es.close();
      status(`Live Whisper is not available (${esc(msg)}). Showing Whisper's saved transcript instead.`);
      setSource("saved", true);
    });
  }

  function setSource(s, keepStatus) {
    src = s;
    $$(".seg button[data-src]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.src === s)));
    if (!C) return;
    closeStream();
    for (const r of C.rows) { r.whisper = s === "saved" ? r.saved : null; fillRow(r); }
    stats();
    if (!keepStatus) status(s === "live" ? "Press play: Whisper transcribes the audio as it plays." : "");
    if (s === "live" && !audio.paused) startStream();
  }
  $$(".seg button[data-src]").forEach((b) => b.addEventListener("click", () => { if (b.dataset.src !== src) setSource(b.dataset.src); }));

  // ---- player ----------------------------------------------------------------------------------
  const setIcon = () => $("#play-icon").setAttribute("d", audio.paused ? "M8 5v14l11-7z" : "M7 5h4v14H7zM13 5h4v14h-4z");
  audio.addEventListener("play", setIcon);
  audio.addEventListener("pause", setIcon);
  audio.addEventListener("loadedmetadata", () => { seek.max = String(audio.duration || C?.duration || 1); });
  playBtn.addEventListener("click", () => {
    if (!C) return;
    stopAt = null;
    if (audio.paused) {
      audio.play().catch(() => status("The browser blocked playback. Press play again."));
      if (src === "live" && !streamStarted) startStream();
    } else audio.pause();
  });
  seek.addEventListener("input", () => { audio.currentTime = +seek.value; stopAt = null; });
  rowsEl.addEventListener("click", (e) => {
    const t = e.target.closest(".w.flag, .gap");
    if (!t) return;
    const a = +t.dataset.t, b = +(t.dataset.t1 || a);
    audio.currentTime = Math.max(0, a - PAD);
    stopAt = b + PAD;
    audio.play();
  });
  document.addEventListener("keydown", (e) => {
    if (mode === "review" && reviewKey(e)) return;
    if (e.code === "Space" && !["SELECT", "INPUT", "BUTTON"].includes(document.activeElement?.tagName)) { e.preventDefault(); playBtn.click(); }
  });

  // ---- human review: the flagged spots one at a time, riskiest first -----------------------------
  // spot: {row, kind: "word"|"gap", from, to, after, g, score, t0, t1, known, wrong}
  let mode = "watch", spots = [], R = null;
  function buildSpots() {
    spots = [];
    for (const row of C.rows) {
      const ws = row.ours.words;
      let cur = null;
      ws.forEach((w, k) => {
        if ((w.score ?? 0) >= TH) {
          if (cur) { cur.to = k; cur.t1 = w.t1; cur.score = Math.max(cur.score, w.score); }
          else cur = { row, kind: "word", from: k, to: k, t0: w.t0, t1: w.t1, score: w.score };
        } else if (cur) { spots.push(cur); cur = null; }
      });
      if (cur) spots.push(cur);
      for (const g of row.ours.gaps || []) if ((g.score ?? 0) >= TH) {
        const t = g.t ?? row.start;
        spots.push({ row, kind: "gap", after: g.after, g, t0: t, t1: t, score: g.score });
      }
    }
    for (const sp of spots) {
      if (sp.kind === "word") {
        const ws = sp.row.ours.words.slice(sp.from, sp.to + 1);
        sp.known = ws.some((w) => w.wrong !== null && w.wrong !== undefined);
        sp.wrong = ws.some((w) => w.wrong === 1);
      } else {
        sp.known = sp.g.truth !== null && sp.g.truth !== undefined;
        sp.wrong = (sp.g.truth ?? 0) > 0;
      }
    }
    spots.sort((a, b) => b.score - a.score);
    R = { i: 0, decided: null, fixing: false, listened: 0, caught: 0, missed: 0, confirmed: 0, falseFix: 0, fixes: 0 };
    $("#nspots").textContent = spots.length ? `· ${spots.length}` : "";
  }

  function setMode(m) {
    mode = m;
    $$("#mode button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === m)));
    $("#review").hidden = m !== "review";
    $("#cols").hidden = m !== "watch";
    $("#srcseg").hidden = m !== "watch";
    $(".legend").hidden = m !== "watch";
    if (m !== "watch") { audio.pause(); closeStream(); status(""); }
    if (m === "review") renderReview(true);
    else if (C) status(src === "live" ? "Press play: Whisper transcribes the audio as it plays." : "Press play.");
  }
  $$("#mode button").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.mode !== mode && C) setMode(b.dataset.mode);
  }));

  function clip(sp) {
    const a = Math.max(sp.row.start, sp.t0 - PAD), b = Math.min(sp.row.end, sp.t1 + PAD);
    audio.currentTime = a;
    stopAt = Math.max(b, a + 0.8);
    R.listened += stopAt - a;
    audio.play().catch(() => {});
  }

  function sentence(sp) {
    const ws = sp.row.ours.words, out = [];
    if (sp.kind === "gap" && sp.after === -1) out.push(`<mark class="caret">‸</mark>`);
    ws.forEach((w, k) => {
      const t = esc(w.fixed !== undefined ? w.fixed : w.text);
      out.push(sp.kind === "word" && k >= sp.from && k <= sp.to ? `<mark>${t}</mark>` : t);
      if (sp.kind === "gap" && sp.after === k) out.push(`<mark class="caret">‸</mark>`);
    });
    return out.join(" ");
  }

  function verdict(sp, d) {
    if (!sp.known) return { cls: "", text: "No human transcript for this spot, so there is no answer to check." };
    if (d === "fix") return sp.wrong ? { cls: "good", text: "Good catch! The model got this wrong." }
                                     : { cls: "meh", text: "The model actually had this right." };
    if (d === "ok") return sp.wrong ? { cls: "bad", text: "Missed: this one was really wrong. Easy to miss." }
                                    : { cls: "good", text: "Right, the model had it correct." };
    return sp.wrong ? { cls: "bad", text: "Skipped, and it was wrong." } : { cls: "", text: "Skipped. The model had it right." };
  }

  function renderReview(play) {
    const el = $("#review");
    if (!spots.length) { el.innerHTML = `<div class="rv-card"><h3>Nothing to review</h3><p>The verifier flagged no spots in this consultation.</p></div>`; return; }
    if (R.i >= spots.length) return renderSummary();
    const sp = spots[R.i], row = sp.row;
    const wh = row.saved ? row.saved.words.map((w) => esc(w.text)).join(" ") : "";
    const pct = Math.round(100 * R.i / spots.length);
    const cur = sp.kind === "word" ? sp.row.ours.words.slice(sp.from, sp.to + 1).map((w) => w.text).join(" ") : "";
    let body;
    if (R.decided) {
      const v = verdict(sp, R.decided);
      body = `<div class="rv-reveal"><span class="chip ${v.cls}">${v.text}</span>
                ${row.ours.reference ? `<p class="rv-truth"><b>What was really said</b> ${esc(row.ours.reference)}</p>` : ""}
                <button class="b-main" type="button" data-act="next">${R.i + 1 < spots.length ? "Next spot" : "See my score"} <kbd>⏎</kbd></button></div>`;
    } else if (R.fixing) {
      body = `<form class="rv-fix" id="rv-fix"><input id="rv-input" autocomplete="off" aria-label="Correct words"
                value="${esc(cur)}" placeholder="${sp.kind === "gap" ? "type the missing words" : ""}">
                <button class="b-main" type="submit">Save fix <kbd>⏎</kbd></button>
                <button class="b-ghost" type="button" data-act="cancel">Cancel</button></form>`;
    } else {
      body = `<div class="rv-acts">
                <button class="b-ghost" type="button" data-act="replay">↻ Replay <kbd>R</kbd></button>
                <button class="b-main" type="button" data-act="ok">${sp.kind === "gap" ? "Nothing missing" : "Sounds right"} <kbd>Y</kbd></button>
                <button class="b-alt" type="button" data-act="fix">${sp.kind === "gap" ? "Add words" : "Fix it"} <kbd>F</kbd></button>
                <button class="b-ghost" type="button" data-act="skip">Skip <kbd>S</kbd></button></div>`;
    }
    el.innerHTML = `<div class="rv-card">
      <div class="rv-top mono"><span>Spot ${R.i + 1} of ${spots.length} · riskiest first</span>
        <span>listened ${fmt(R.listened)} of ${fmt(C.duration)}</span></div>
      <div class="rv-bar"><i style="width:${pct}%"></i></div>
      <span class="who">${esc(row.role)} · ${fmt(row.start)}</span>
      <p class="rv-q">${sp.kind === "gap" ? "Are words missing at ‸ ?" : "Is the highlighted word right?"}</p>
      <p class="rv-sent">${sentence(sp)}</p>
      ${wh ? `<p class="rv-wh"><b>Whisper heard</b> ${wh}</p>` : ""}
      ${body}
      <div class="rv-score mono"><span>caught <b>${R.caught}</b></span><span>confirmed <b>${R.confirmed}</b></span>
        <span>missed <b>${R.missed}</b></span><button class="b-ghost" type="button" data-act="finish">Finish</button></div>
    </div>`;
    if (R.fixing) { const inp = $("#rv-input"); inp.focus(); inp.select(); }
    if (play) clip(sp);
  }

  function decide(d, text) {
    const sp = spots[R.i];
    R.decided = d; R.fixing = false;
    if (d === "fix") {
      R.fixes++;
      if (sp.kind === "word") {
        const ws = sp.row.ours.words;
        for (let k = sp.from; k <= sp.to; k++) ws[k].fixed = k === sp.from ? text : "";
      } else if (text) sp.row.gapFix.set(sp.after, text);
      else sp.g.ok = true;
      if (sp.known) { if (sp.wrong) R.caught++; else R.falseFix++; }
    } else if (d === "ok") {
      if (sp.kind === "word") sp.row.ours.words.slice(sp.from, sp.to + 1).forEach((w) => { w.ok = true; });
      else sp.g.ok = true;
      if (sp.known) { if (sp.wrong) R.missed++; else R.confirmed++; }
    } else if (sp.known && sp.wrong) R.missed++;
    fillRow(sp.row);
    renderReview(false);
  }

  const n = (k, one, many = `${one}s`) => `${k} ${k === 1 ? one : many}`;
  function renderSummary() {
    audio.pause();
    const real = spots.filter((s) => s.known && s.wrong).length;
    const share = C.duration ? Math.round(100 * R.listened / C.duration) : 0;
    $("#review").innerHTML = `<div class="rv-card rv-end">
      <h3>Review done</h3>
      <div class="rv-big">
        <div><b class="mono">${R.caught}<small>/${real}</small></b><span>real errors you caught</span></div>
        <div><b class="mono">${fmt(R.listened)}</b><span>listened, instead of the full ${fmt(C.duration)} (${share}%)</span></div>
        <div><b class="mono">${R.fixes}</b><span>fixes written into the transcript</span></div>
      </div>
      <p>${n(R.confirmed, "spot")} you confirmed ${R.confirmed === 1 ? "was" : "were"} right, ${n(R.missed, "real error")} slipped through${R.falseFix ? `, ${n(R.falseFix, "fix", "fixes")} changed words that were already right` : ""}.</p>
      <div class="rv-acts"><button class="b-main" type="button" data-act="download">Download corrected transcript</button>
        <button class="b-alt" type="button" data-act="watch">See it in Watch</button>
        <button class="b-ghost" type="button" data-act="again">Start over</button></div>
    </div>`;
  }

  function download() {
    const lines = C.rows.map((r) => {
      const parts = [];
      if (r.gapFix.has(-1)) parts.push(r.gapFix.get(-1));
      r.ours.words.forEach((w, k) => {
        const t = w.fixed !== undefined ? w.fixed : w.text;
        if (t) parts.push(t);
        if (r.gapFix.has(k)) parts.push(r.gapFix.get(k));
      });
      return `[${fmt(r.start)}] ${String(r.role).toUpperCase()}: ${parts.join(" ")}`;
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/plain" }));
    a.download = `${C.id}_reviewed.txt`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function resetReview() {
    for (const r of C.rows) {
      r.gapFix.clear();
      r.ours.words.forEach((w) => { delete w.fixed; delete w.ok; });
      (r.ours.gaps || []).forEach((g) => { delete g.ok; });
      fillRow(r);
    }
    buildSpots();
  }

  function act(a) {
    const sp = spots[R.i];
    if (a === "replay" && sp) clip(sp);
    else if (a === "ok" || a === "skip") decide(a);
    else if (a === "fix") { R.fixing = true; renderReview(false); }
    else if (a === "cancel") { R.fixing = false; renderReview(false); }
    else if (a === "next") { R.i++; R.decided = null; renderReview(true); }
    else if (a === "finish") { R.i = spots.length; renderReview(false); }
    else if (a === "download") download();
    else if (a === "watch") setMode("watch");
    else if (a === "again") { resetReview(); renderReview(true); }
  }
  $("#review").addEventListener("click", (e) => { const b = e.target.closest("[data-act]"); if (b) act(b.dataset.act); });
  $("#review").addEventListener("submit", (e) => {
    e.preventDefault();
    const sp = spots[R.i], text = $("#rv-input").value.trim().replace(/\s+/g, " ");
    const before = sp.kind === "word" ? sp.row.ours.words.slice(sp.from, sp.to + 1).map((w) => w.text).join(" ") : "";
    decide(text === before ? "ok" : "fix", text);   // saving the words unchanged means they were right
  });

  function reviewKey(e) {
    if (e.target.tagName === "INPUT") { if (e.key === "Escape") act("cancel"); return e.key === "Escape"; }
    if (!R || R.i >= spots.length || e.ctrlKey || e.metaKey || e.altKey) return false;
    const map = R.decided ? { enter: "next", n: "next", r: "replay" } : { r: "replay", y: "ok", f: "fix", s: "skip" };
    const a = map[e.key.toLowerCase()];
    if (!a) return false;
    e.preventDefault();
    act(a);
    return true;
  }

  // ---- consultations ---------------------------------------------------------------------------
  async function open(id) {
    audio.pause(); closeStream(); stopAt = null;
    if (mode !== "watch") setMode("watch");
    status(`Loading ${esc(id)}…`);
    try {
      const [ours, whisper] = await Promise.all([
        getJSON(`/api/v2/consultation/${enc(id)}?unit=turn&model=${OURS}`),
        getJSON(`/api/v2/consultation/${enc(id)}?unit=turn&model=${WHISPER}`).catch(() => null),
      ]);
      C = { id, duration: ours.duration };
      build(ours, whisper);
      audio.src = `/api/v2/consultation/${enc(id)}/audio`;
      audio.currentTime = 0;
      seek.max = String(ours.duration || 1); seek.value = "0"; seek.disabled = false; playBtn.disabled = false;
      status(src === "live" ? "Press play: Whisper transcribes the audio as it plays." : "Press play.");
    } catch (err) {
      status(`Could not load ${esc(id)}: ${esc(err.message)}`);
    }
  }
  $("#consult").addEventListener("change", (e) => open(e.target.value));

  async function init() {
    requestAnimationFrame(tick);
    fetch("/api/warmup", { method: "POST" }).catch(() => {});   // load Whisper now, not on the first play
    try {
      const m = await getJSON("/api/v2/models");
      if (!m.models.find((x) => x.id === OURS && x.available)) {
        status("Our model's results are missing. Run <code>python experiments/lora_pilot/export_app.py</code> first.");
        return;
      }
      const list = await getJSON(`/api/v2/consultations?model=${OURS}`);
      list.sort((a, b) => (a.split !== "test") - (b.split !== "test") || (b.has_audio - a.has_audio) || a.id.localeCompare(b.id));
      const sel = $("#consult");
      sel.innerHTML = list.map((c) => `<option value="${esc(c.id)}"${c.has_audio ? "" : " disabled"}>` +
        `${esc(c.id.replace("_", " · "))} · ${fmt(c.duration)}${c.split === "test" ? "" : " · used in training"}${c.has_audio ? "" : " · no audio"}</option>`).join("");
      sel.disabled = false;
      const first = list.find((c) => c.has_audio && c.split === "test") || list[0];
      sel.value = first.id;
      await open(first.id);
    } catch (err) {
      status(`Could not reach the server (${esc(err.message)}). Start it with <code>uvicorn app.server:app --port 8765</code>.`);
    }
  }
  init();
})();
