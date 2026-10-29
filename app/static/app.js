// Second Listen: review a Whisper transcript by listening only to the words and gaps the verifier flags.
// Data comes from the /api/v2 endpoints (app/server.py); open with ?mock=1 to use the fixtures in mock/.
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const fmt = (s) => { s = Math.max(0, Math.round(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const enc = encodeURIComponent;
  const PAD = 0.6;  // seconds of context around a flagged clip
  const MOCK = new URLSearchParams(location.search).has("mock");

  // ---- API -------------------------------------------------------------------------------------
  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) {
      let msg = `${r.status} ${r.statusText}`;
      try { const j = await r.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) { /* not JSON */ }
      throw new Error(msg);
    }
    return r.json();
  }
  const API = {
    consultations: () => getJSON(MOCK ? "mock/consultations.json" : "/api/v2/consultations"),
    payload: (id) => getJSON(MOCK ? `mock/consultation_${id}.json` : `/api/v2/consultation/${enc(id)}?unit=turn`),
    audioUrl: (id) => (MOCK ? `mock/${id}.mp3` : `/api/v2/consultation/${enc(id)}/audio`),
    curves: () => getJSON(MOCK ? "mock/curves.json" : "/api/v2/curves"),
    live: (id, second) => (MOCK ? new MockSource(mockPayload, 2.5) : new EventSource(`/api/v2/live/${enc(id)}?second=${second ? 1 : 0}`)),
    async upload(file) {
      const fd = new FormData(); fd.append("file", file);
      const r = await fetch("/api/v2/upload", { method: "POST", body: fd });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json();
    },
    jobEvents: (job) => new EventSource(`/api/v2/jobs/${enc(job)}/events`),
  };

  // Mock stand-in for EventSource: replays a saved payload as if Whisper were decoding it at `speed`× real time.
  let mockPayload = null;
  class MockSource {
    constructor(payload, speed) {
      this.handlers = {}; this.closed = false;
      const units = [...(payload?.units || [])].sort((a, b) => a.end - b.end);
      const dur = payload?.duration || 0;
      const t0 = performance.now();
      let i = 0;
      this.timer = setInterval(() => {
        if (this.closed) return;
        const decoded = ((performance.now() - t0) / 1000) * speed;
        while (i < units.length && units[i].end <= decoded) {
          const u = units[i++];
          this.emit("turn", { ...u, reference: undefined, words: u.words.map(({ wrong, ...w }) => w), gaps: u.gaps.map(({ truth, ...g }) => g), decode_s: 0.4 });
        }
        this.emit("progress", { decoded_until: Math.min(decoded, dur), realtime_factor: 1 / speed });
        if (i >= units.length) { this.emit("done", {}); this.close(); }
      }, 250);
    }
    addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
    emit(type, obj) { (this.handlers[type] || []).forEach((fn) => fn({ data: JSON.stringify(obj) })); }
    close() { this.closed = true; clearInterval(this.timer); }
  }

  // ---- session: the transcript currently on screen ----------------------------------------------
  // unit: {id, role, s, e, ref, wi: [word indices], gapList: [{after, at, score, truth, txt}]}
  // word: {ui, text, p, crit, v3, wrong, t0, t1}; wrong / truth are null when there is no human transcript
  let S = { kind: "none", id: null, split: null, duration: 0, units: [] };
  let W = [];

  function roleClass(role) { return String(role || "speaker").replace(/[^a-z]+/gi, "-").toLowerCase(); }

  function addUnit(raw) {
    const words = raw.words || [];
    const s = +raw.start, e = +raw.end;
    const ui = S.units.length;
    const u = { id: raw.id, role: raw.role || "speaker", s, e, ref: raw.reference ?? S.refs?.get(raw.id) ?? null, wi: [], gapList: [] };
    // spread words evenly over the unit when the server sends no word times
    const lens = words.map((w) => String(w.text).length + 1);
    const tot = lens.reduce((a, b) => a + b, 0) || 1;
    const s0 = s + 0.1, span = Math.max(0.2, e - 0.1 - s0);
    let acc = 0;
    words.forEach((w, k) => {
      const a0 = s0 + span * acc / tot; acc += lens[k];
      const a1 = s0 + span * acc / tot;
      u.wi.push(W.length);
      W.push({ ui, k, text: w.text, p: w.p ?? null, crit: !!w.critical, v3: w.score ?? 0, wrong: w.wrong ?? null,
               t0: Number.isFinite(w.t0) ? w.t0 : a0, t1: Number.isFinite(w.t1) ? w.t1 : a1 });
    });
    const byAfter = new Map();
    for (const g of raw.gaps || []) {
      const m = byAfter.get(g.after) || { after: g.after, score: 0, truth: null, txt: [], at: null };
      m.score = Math.max(m.score, g.score ?? 0);
      if (g.truth !== undefined && g.truth !== null) m.truth = (m.truth ?? 0) + g.truth;
      if (g.suggestion) m.txt.push(g.suggestion);
      if (Number.isFinite(g.t)) m.at = g.t;
      byAfter.set(g.after, m);
    }
    u.gapList = [...byAfter.values()].map((g) => ({
      ...g, ui, txt: g.txt.join(" "),
      at: g.at ?? (g.after < 0 || !u.wi.length ? s0 : W[u.wi[Math.min(g.after, u.wi.length - 1)]].t1),
    }));
    S.units.push(u);
    return ui;
  }

  function newSession(kind, info = {}) {
    closeStream();
    S = { kind, id: info.id ?? null, split: info.split ?? null, duration: info.duration || 0, units: [], name: info.name };
    W = [];
    state.decisions = new Map(); state.sel = null;
    shownWords = new Uint8Array(0);
  }

  const state = { th: 0.25, crit: true, view: null, dom: "window", err: "weighted", budget: 60, showRef: false, sel: null,
                  decisions: new Map(), mode: "replay", curves: null };
  const isFlag = (w) => w.v3 >= state.th || (state.crit && w.crit);
  const gapFlag = (g) => g.score >= state.th;

  function spans() {
    const out = [];
    S.units.forEach((t, ui) => {
      let cur = null;
      for (const wi of t.wi) {
        const w = W[wi];
        if (isFlag(w)) {
          if (cur) { cur.to = wi; cur.t1 = w.t1; cur.c = cur.c || w.crit; }
          else cur = { type: "word", ui, from: wi, to: wi, t0: w.t0, t1: w.t1, c: w.crit };
        } else if (cur) { out.push(cur); cur = null; }
      }
      if (cur) out.push(cur);
      for (const g of t.gapList) if (gapFlag(g)) out.push({ type: "gap", ui, g, t0: g.at, t1: g.at, after: g.after });
    });
    for (const s of out) {
      const t = S.units[s.ui];
      s.c0 = Math.max(t.s, s.t0 - PAD); s.c1 = Math.min(t.e, s.t1 + PAD);
      if (s.type === "word") {
        s.key = `w${s.from}-${s.to}`;
        const ws = W.slice(s.from, s.to + 1);
        s.known = ws.some((w) => w.wrong !== null);
        s.wrong = ws.some((w) => w.wrong === 1);
      } else {
        s.key = `g${s.ui}:${s.after}`;
        s.known = s.g.truth !== null;
        s.wrong = (s.g.truth ?? 0) > 0;
      }
    }
    out.sort((a, b) => a.t0 - b.t0 || (a.type === "gap") - (b.type === "gap"));
    return out;
  }
  function listenTime(list, upTo = Infinity) {
    const iv = list.filter((s) => s.t0 <= upTo).map((s) => [s.c0, Math.min(s.c1, upTo)]).sort((a, b) => a[0] - b[0]);
    let total = 0, end = -1;
    for (const [a, b] of iv) { if (b <= end) continue; total += b - Math.max(a, end); end = b; }
    return total;
  }

  // ---- notices ---------------------------------------------------------------------------------
  function notice(html, err = false) {
    const n = $("#notice");
    n.hidden = !html; n.innerHTML = html || ""; n.classList.toggle("err", !!err);
  }
  function audioNote(html) { const n = $("#audio-note"); n.hidden = !html; n.innerHTML = html || ""; }

  // ---- clock: the audio element, or a timer when there is no audio --------------------------------
  const audio = $("#audio");
  let hasAudio = false, audioUrl = null, peaks = null;
  const virt = { t: 0, playing: false, last: 0, rate: 1 };
  let clipEnd = null;
  const clock = {
    get now() { return hasAudio ? (audio.currentTime || 0) : virt.t; },
    get paused() { return hasAudio ? audio.paused : !virt.playing; },
    play() {
      if (hasAudio) return audio.play().catch(() => {});
      if (virt.t >= dur() - 0.05) virt.t = 0;
      virt.playing = true; virt.last = performance.now(); setPlayIcon(); loop();
    },
    pause() { if (hasAudio) audio.pause(); else { virt.playing = false; setPlayIcon(); } },
    seek(t) { t = Math.min(Math.max(0, dur() - 0.05), Math.max(0, t)); if (hasAudio) audio.currentTime = t; else virt.t = t; },
    set rate(r) { audio.playbackRate = r; virt.rate = r; },
  };
  const dur = () => S.duration || (hasAudio && Number.isFinite(audio.duration) ? audio.duration : 0) ||
    Math.max(0, ...S.units.map((u) => u.e));

  const ICON_PLAY = '<svg viewBox="0 0 24 24"><path d="M7 4.5v15l12.5-7.5z" fill="currentColor"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 24 24"><path d="M6.5 4.5h4v15h-4zM13.5 4.5h4v15h-4z" fill="currentColor"/></svg>';
  const playBtn = $("#play");
  function setPlayIcon() { const p = !clock.paused; playBtn.innerHTML = p ? ICON_PAUSE : ICON_PLAY; playBtn.setAttribute("aria-label", p ? "Pause" : "Play"); }
  setPlayIcon();
  playBtn.addEventListener("click", () => {
    clipEnd = null;
    if (clock.paused) { waitingForDecode = false; clock.play(); } else clock.pause();
  });
  audio.addEventListener("play", () => { setPlayIcon(); loop(); });
  audio.addEventListener("pause", setPlayIcon);
  audio.addEventListener("ended", setPlayIcon);
  audio.addEventListener("seeked", () => tick(true));
  audio.addEventListener("loadedmetadata", () => { $("#t-dur").textContent = fmt(dur()); });
  $$(".speeds button").forEach((b) => b.addEventListener("click", () => {
    clock.rate = +b.dataset.rate;
    $$(".speeds button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  }));

  let raf = 0;
  function loop() {
    cancelAnimationFrame(raf);
    const f = (ts) => {
      if (!hasAudio && virt.playing) {
        virt.t += ((ts - virt.last) / 1000) * virt.rate; virt.last = ts;
        if (virt.t >= dur()) { virt.t = dur(); virt.playing = false; setPlayIcon(); }
      }
      tick();
      if (!clock.paused) raf = requestAnimationFrame(f);
    };
    virt.last = performance.now();
    raf = requestAnimationFrame(f);
  }

  // Load audio for playback and compute waveform peaks from it. Returns false when there is none.
  async function loadAudio(url) {
    releaseAudio();
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(String(r.status));
      const buf = await r.arrayBuffer();
      return await useAudioBuffer(buf, r.headers.get("content-type") || "audio/wav");
    } catch (_) {
      return false;
    }
  }
  async function useAudioBuffer(buf, type) {
    audioUrl = URL.createObjectURL(new Blob([buf], { type }));
    audio.src = audioUrl;
    hasAudio = true;
    peaks = null;
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      const decoded = await ctx.decodeAudioData(buf.slice(0));
      ctx.close();
      const ch = decoded.getChannelData(0), n = 900, step = Math.max(1, Math.floor(ch.length / n));
      const out = new Float32Array(n);
      let mx = 1e-9;
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = i * step, end = Math.min(ch.length, (i + 1) * step); j < end; j++) sum += ch[j] * ch[j];
        out[i] = Math.sqrt(sum / step); mx = Math.max(mx, out[i]);
      }
      for (let i = 0; i < n; i++) out[i] /= mx;
      peaks = out;
      if (!S.duration) S.duration = decoded.duration;
    } catch (_) { /* playback still works; the waveform falls back to turn blocks */ }
    return true;
  }
  function releaseAudio() {
    audio.pause();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = null; hasAudio = false; peaks = null; virt.t = 0; virt.playing = false;
    audio.removeAttribute("src"); audio.load();
    setPlayIcon();
  }

  // ---- waveform --------------------------------------------------------------------------------
  const cv = $("#wave"), cx = cv.getContext("2d");
  let currentSpans = [];
  function coverage(n, D) {
    // which speaker's turns cover each waveform bin (1 doctor, 2 patient, 3 both)
    const cov = new Uint8Array(n);
    for (const u of S.units) {
      const bit = u.role === "patient" ? 2 : u.role === "doctor" ? 1 : 3;
      const a = Math.floor((u.s / D) * n), b = Math.min(n - 1, Math.ceil((u.e / D) * n));
      for (let i = Math.max(0, a); i <= b; i++) cov[i] |= bit;
    }
    return cov;
  }
  function drawWave() {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (!w) return;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cx.clearRect(0, 0, w, h);
    const D = dur();
    if (!D) return;
    const now = clock.now, mid = h / 2 - 4, n = peaks ? peaks.length : 300, bw = w / n;
    const cd = css("--doctor"), cp = css("--patient"), cl = css("--line"), cf = css("--flag"), cr = css("--crit"), ci = css("--ink");
    const cov = coverage(n, D);
    for (let i = 0; i < n; i++) {
      const x = i * bw, played = (i / n) * D <= now;
      cx.globalAlpha = played ? 1 : 0.32;
      const amp = peaks ? Math.max(1, Math.sqrt(peaks[i]) * (mid - 2)) : (cov[i] ? mid * 0.45 : 1);
      cx.fillStyle = cov[i] & 1 ? cd : cl; cx.fillRect(x, mid - amp, Math.max(1, bw - 0.6), amp);
      cx.fillStyle = cov[i] & 2 ? cp : cl; cx.fillRect(x, mid + 1, Math.max(1, bw - 0.6), amp);
    }
    cx.globalAlpha = 1;
    for (const s of currentSpans) {
      cx.fillStyle = s.type === "gap" ? cr : cf;
      cx.fillRect((s.c0 / D) * w, h - 5, Math.max(2, ((s.t1 - s.t0) / D) * w + 2), 4);
    }
    cx.fillStyle = ci; cx.fillRect((now / D) * w - 1, 0, 2, h - 6);
  }
  cv.addEventListener("click", (e) => {
    const r = cv.getBoundingClientRect();
    clipEnd = null; waitingForDecode = false;
    clock.seek(((e.clientX - r.left) / r.width) * dur());
    tick(true);
  });
  cv.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault(); clipEnd = null;
      clock.seek(clock.now + (e.key === "ArrowRight" ? 5 : -5));
      tick(true);
    }
  });
  new ResizeObserver(drawWave).observe(cv);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawWave(); drawChart(); });
  new MutationObserver(() => { drawWave(); drawChart(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  // ---- live view -------------------------------------------------------------------------------
  const stream = $("#stream"), behind = $("#behind");
  let shownWords = new Uint8Array(0);
  let liveGaps = [];
  function wordHTML(w, wi) {
    const cls = ["w"];
    if (w.crit) cls.push("crit");
    else if (w.v3 >= state.th) cls.push("low");
    const a = Math.round(14 + Math.min(1, w.v3) * 40);
    const tip = `expected weighted error ${w.v3.toFixed(2)}` + (w.p !== null ? ` · Whisper ${Math.round(w.p * 100)}% sure` : "");
    return `<span class="${cls.join(" ")} hidden" id="w${wi}" style="--a:${a}%" title="${tip}">${esc(w.text)}</span>`;
  }
  function gapHTML(g, id) {
    const tip = g.txt ? `possibly missing: ${g.txt}` : "words may be missing here";
    return `<span class="gap hidden" id="${id}" title="${esc(tip)}" aria-label="${esc(tip)}">‸</span>`;
  }
  function unitEl(ui) {
    const t = S.units[ui];
    const gapsAfter = new Map(t.gapList.filter(gapFlag).map((g) => [g.after, g]));
    const parts = [];
    const addGap = (after) => {
      const g = gapsAfter.get(after); if (!g) return;
      const id = `gp${ui}_${after}`; liveGaps.push({ id, g, on: false }); parts.push(gapHTML(g, id));
    };
    addGap(-1);
    t.wi.forEach((wi, k) => { parts.push(wordHTML(W[wi], wi)); addGap(k); });
    const el = document.createElement("div");
    el.className = `turn ${roleClass(t.role)}`; el.id = `t${ui}`; el.hidden = true; el.dataset.s = t.s;
    el.innerHTML = `<div class="who">${esc(t.role)}<span class="mono">${fmt(t.s)}</span></div>` +
      `<div class="bubble">${parts.join(" ")}</div>` +
      (t.ref !== null ? `<p class="ref" ${state.showRef ? "" : "hidden"}>${esc(t.ref)}</p>` : "");
    return el;
  }
  function placeUnit(ui) {
    // keep bubbles in time order even when turns arrive out of order
    const el = unitEl(ui), s = S.units[ui].s;
    const next = [...stream.querySelectorAll(".turn")].find((n) => +n.dataset.s > s);
    stream.insertBefore(el, next || behind);
    if (shownWords.length < W.length) { const a = new Uint8Array(W.length); a.set(shownWords); shownWords = a; }
  }
  function buildLive() {
    stream.querySelectorAll(".turn").forEach((n) => n.remove());
    shownWords = new Uint8Array(W.length);
    liveGaps = [];
    S.units.forEach((_, ui) => placeUnit(ui));
  }
  function renderLive(now, jump) {
    const nearBottom = stream.scrollHeight - stream.scrollTop - stream.clientHeight < 80;
    let added = false, words = 0, flags = 0;
    S.units.forEach((t, ui) => {
      const el = document.getElementById(`t${ui}`);
      if (!el) return;
      const vis = t.s <= now;
      if (el.hidden === vis) { el.hidden = !vis; added = added || vis; }
      el.classList.toggle("speaking", vis && now < t.e && !clock.paused);
      for (const wi of t.wi) {
        const w = W[wi], on = w.t0 <= now;
        if (on) { words++; if (isFlag(w)) flags++; }
        if (on !== !!shownWords[wi]) {
          shownWords[wi] = on ? 1 : 0;
          const n = document.getElementById(`w${wi}`);
          if (n) { n.classList.toggle("hidden", !on); if (on && !jump) { n.classList.add("fresh"); added = true; } }
        }
      }
    });
    for (const lg of liveGaps) {
      const on = lg.g.at <= now;
      if (on) flags++;
      if (on !== lg.on) {
        lg.on = on;
        const n = document.getElementById(lg.id);
        if (n) { n.classList.toggle("hidden", !on); if (on && !jump) n.classList.add("fresh"); }
      }
    }
    const streaming = live.active && !live.done;
    behind.hidden = !(streaming && now > live.decodedUntil + 0.25);
    if (!behind.hidden) {
      behind.lastElementChild.textContent = `Decoding… Whisper is ${Math.round(now - live.decodedUntil)} s behind the audio`;
      stream.appendChild(behind);
    }
    $("#empty").hidden = now > 0.05 || words > 0 || streaming;
    $("#s-words").textContent = words;
    $("#s-flags").textContent = flags;
    const lt = listenTime(currentSpans, now);
    $("#s-listen").innerHTML = `${fmt(lt)} <small>of ${fmt(now)}</small>`;
    $("#s-listen-bar").style.width = `${now > 0 ? (100 * lt / now) : 0}%`;
    if ((added || !behind.hidden) && (nearBottom || jump)) stream.scrollTop = stream.scrollHeight;
  }
  $("#show-ref").addEventListener("change", (e) => {
    state.showRef = e.target.checked;
    $$("#stream .ref").forEach((p) => (p.hidden = !state.showRef));
  });

  let waitingForDecode = false;
  function tick(jump = false) {
    const now = clock.now;
    if (clipEnd !== null && now >= clipEnd) { clock.pause(); clipEnd = null; }
    // keep the audio in step with a live decode that is slower than real time
    if (live.active && !live.done && $("#sync").checked && !clock.paused && now > live.decodedUntil + 0.5) {
      clock.pause(); waitingForDecode = true;
    }
    $("#t-now").textContent = fmt(now);
    $("#t-dur").textContent = fmt(dur());
    cv.setAttribute("aria-valuenow", String(Math.round(now)));
    cv.setAttribute("aria-valuemax", String(Math.round(dur())));
    if (state.view === "live") renderLive(now, jump);
    drawWave();
  }

  // ---- live decoding (Run Whisper now / upload) --------------------------------------------------
  const live = { active: false, done: false, decodedUntil: 0, source: null, t0: 0 };
  function closeStream() {
    if (live.source) { try { live.source.close(); } catch (_) { /* already closed */ } }
    Object.assign(live, { active: false, done: false, decodedUntil: 0, source: null });
    waitingForDecode = false;
    behind.hidden = true;
  }
  function liveStatus(html, kind = "") {
    const n = $("#live-status");
    n.className = `status ${kind}`;
    n.innerHTML = html;
  }
  function attachStream(src, label) {
    live.source = src; live.active = true; live.done = false; live.decodedUntil = 0; live.t0 = performance.now();
    liveStatus(`<span class="spinner" aria-hidden="true"></span>${esc(label)}: starting Whisper…`);
    src.addEventListener("turn", (e) => {
      const ui = addUnit(JSON.parse(e.data));
      live.decodedUntil = Math.max(live.decodedUntil, S.units[ui].e);
      currentSpans = spans();
      if (state.view === "live") placeUnit(ui);
      if (state.view === "review") buildDoc();
      resumeIfCaughtUp();
      tick();
    });
    src.addEventListener("progress", (e) => {
      const d = JSON.parse(e.data);
      live.decodedUntil = Math.max(live.decodedUntil, d.decoded_until || 0);
      const rtf = d.realtime_factor ? ` · ${(+d.realtime_factor).toFixed(2)}× real time` : "";
      liveStatus(`<span class="spinner" aria-hidden="true"></span>${esc(label)}: decoded ${fmt(live.decodedUntil)} of ${fmt(dur())}${rtf}`);
      resumeIfCaughtUp();
      tick();
    });
    src.addEventListener("done", () => {
      live.done = true;
      const secs = ((performance.now() - live.t0) / 1000).toFixed(0);
      liveStatus(`${esc(label)}: finished, ${S.units.length} ${S.units.length === 1 ? "segment" : "segments"} in ${secs} s.`);
      src.close();
      resumeIfCaughtUp();
      tick();
    });
    // a server "error" event carries data; a bare error is the connection dropping
    src.addEventListener("error", (e) => {
      if (live.done) return;
      let msg = "The connection to the server was lost.";
      if (e && e.data) { try { msg = JSON.parse(e.data).message || msg; } catch (_) { msg = String(e.data); } }
      live.done = true;
      liveStatus(`Whisper stopped: ${esc(msg)}`, "err");
      src.close();
      tick();
    });
  }
  function resumeIfCaughtUp() {
    if (waitingForDecode && (live.done || live.decodedUntil > clock.now + 2)) { waitingForDecode = false; clock.play(); }
  }

  function setMode(mode) {
    state.mode = mode;
    $$(".seg button[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    const run = mode === "run";
    $("#second-wrap").hidden = !run || S.kind === "upload";
    $("#sync-wrap").hidden = !run;
    $("#run-start").hidden = !run || S.kind === "upload";
  }
  $$(".seg button[data-mode]").forEach((b) => b.addEventListener("click", async () => {
    if (b.dataset.mode === state.mode && S.kind !== "upload") return;
    clock.pause(); clock.seek(0);
    if (b.dataset.mode === "replay") {
      setMode("replay");
      await openConsultation(lastConsult, { keepAudio: S.kind !== "upload" });
    } else {
      setMode("run");
      if (S.kind === "upload") await openConsultation(lastConsult, { keepAudio: false, mode: "run" });
      else startRunSession(false);
    }
  }));
  function startRunSession(begin) {
    const info = { id: S.id, split: S.split, duration: S.duration };
    // keep the human transcript of each turn, so a live decode of a known consultation can still be compared
    const refs = S.refs || new Map(S.units.filter((u) => u.ref !== null).map((u) => [u.id, u.ref]));
    newSession("live", info);
    S.refs = refs;
    currentSpans = [];
    liveStatus(begin ? "" : "Press <b>Start transcribing</b>. Whisper decodes each turn as the audio plays.");
    setFine();
    rebuildView();
    tick(true);
  }
  $("#run-start").addEventListener("click", () => {
    if (!S.id) return;
    startRunSession(true);
    clock.seek(0);
    attachStream(API.live(S.id, $("#second").checked), $("#second").checked ? "whisper-small + large-v3-turbo" : "whisper-small");
    if (hasAudio) clock.play();
  });

  $("#upload").addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    if (!file) return;
    if (MOCK) { notice("Uploading needs the Second Listen server. Mock mode only replays the sample consultation.", true); return; }
    clock.pause();
    notice(`Uploading <b>${esc(file.name)}</b>…`);
    try {
      const { job } = await API.upload(file);
      newSession("upload", { id: null, name: file.name });
      releaseAudio();
      await useAudioBuffer(await file.arrayBuffer(), file.type || "audio/wav");
      showSplit();
      selectUploadOption(file.name);
      setMode("run");
      setFine();
      notice("");
      location.hash = "#live";
      rebuildView();
      attachStream(API.jobEvents(job), file.name);
      clock.play();
    } catch (err) {
      notice(`Could not upload <b>${esc(file.name)}</b>: ${esc(err.message)}. Check that the server is running and the file is audio.`, true);
    }
  });
  function selectUploadOption(name) {
    const sel = $("#consult");
    let opt = sel.querySelector('option[value="__upload__"]');
    if (!opt) { opt = document.createElement("option"); opt.value = "__upload__"; sel.prepend(opt); }
    opt.textContent = `Uploaded: ${name}`;
    sel.value = "__upload__";
    $("#meta").innerHTML = `${esc(name)}<br>uploaded audio · no human transcript`;
  }

  // ---- review view -----------------------------------------------------------------------------
  const doc = $("#doc"), card = $("#card");
  function spanBtn(s, inner) {
    const d = state.decisions.get(s.key);
    const cls = ["span", s.type === "gap" ? "g" : (s.c ? "c" : ""), d ? `done-${d}` : "", state.sel === s.key ? "on" : ""].join(" ");
    const lab = s.type === "gap" ? ` aria-label="Possible missing words"` : "";
    return `<button type="button" class="${cls}" data-key="${s.key}"${lab}>${inner}</button>`;
  }
  function buildDoc() {
    currentSpans = spans();
    if (!S.units.length) {
      doc.innerHTML = `<div class="empty"><p>${live.active && !live.done ? "Waiting for Whisper's first turn…" : "No transcript yet. Pick a consultation, or start Whisper on the Live tab."}</p></div>`;
      card.innerHTML = ""; updateProgress(); drawWave(); return;
    }
    const byFrom = new Map(currentSpans.filter((s) => s.type === "word").map((s) => [s.from, s]));
    const gapAt = new Map(currentSpans.filter((s) => s.type === "gap").map((s) => [`${s.ui}:${s.after}`, s]));
    const order = S.units.map((_, i) => i).sort((a, b) => S.units[a].s - S.units[b].s);
    doc.innerHTML = order.map((ui) => {
      const t = S.units[ui], parts = [];
      const g0 = gapAt.get(`${ui}:-1`); if (g0) parts.push(spanBtn(g0, "‸"));
      for (let k = 0; k < t.wi.length; k++) {
        const wi = t.wi[k], s = byFrom.get(wi);
        if (s) {
          const txt = [];
          for (let i = s.from; i <= s.to; i++) txt.push(esc(W[i].text));
          parts.push(spanBtn(s, txt.join(" ")));
          for (let kk = k; kk < k + (s.to - s.from); kk++) { const gg = gapAt.get(`${ui}:${kk}`); if (gg) parts.push(spanBtn(gg, "‸")); }
          k += s.to - s.from;
        } else parts.push(esc(W[wi].text));
        const g = gapAt.get(`${ui}:${k}`); if (g) parts.push(spanBtn(g, "‸"));
      }
      return `<div class="turn ${roleClass(t.role)}"><div class="who">${esc(t.role)}<span class="mono">${fmt(t.s)}</span></div><div class="bubble">${parts.join(" ")}</div></div>`;
    }).join("");
    $$(".span", doc).forEach((b) => b.addEventListener("click", () => select(b.dataset.key, true)));
    if (!currentSpans.find((s) => s.key === state.sel)) state.sel = currentSpans[0]?.key ?? null;
    updateProgress(); renderCard(); drawWave();
  }
  function updateProgress() {
    const done = currentSpans.filter((s) => state.decisions.has(s.key)).length;
    $("#p-done").textContent = done; $("#p-total").textContent = currentSpans.length;
    $("#p-bar").style.width = `${currentSpans.length ? 100 * done / currentSpans.length : 0}%`;
    $("#p-listen").textContent = fmt(listenTime(currentSpans));
    $("#p-dur").textContent = fmt(dur());
  }
  function select(key, play) {
    state.sel = key;
    $$(".span", doc).forEach((b) => b.classList.toggle("on", b.dataset.key === key));
    const b = doc.querySelector(`.span[data-key="${key}"]`);
    if (b) b.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    renderCard();
    if (play) playClip();
  }
  function playClip() {
    const s = currentSpans.find((x) => x.key === state.sel); if (!s || !hasAudio) return;
    clock.seek(s.c0); clipEnd = s.c1; clock.play();
  }
  function verdictHTML(s, d) {
    if (!d) return "";
    if (!s.known) return `<div class="verdict fa"><b>Recorded.</b> There is no human transcript for this audio, so the decision cannot be scored.</div>`;
    const gap = s.type === "gap";
    if (s.wrong && d === "fix") return `<div class="verdict hit"><b>Real error caught.</b> ${gap ? "Scoring says words were dropped here." : "Scoring against the human transcript marks this as wrong."}</div>`;
    if (s.wrong && d === "ok") return `<div class="verdict miss"><b>Missed.</b> ${gap ? "Scoring says words were dropped here." : "Scoring marks a word here as wrong."} Compare with the human transcript above.</div>`;
    if (!s.wrong && d === "ok") return `<div class="verdict hit"><b>Agreed.</b> ${gap ? "Nothing is missing here." : "Whisper had this right."} The flag was a false alarm.</div>`;
    return `<div class="verdict fa"><b>Scoring disagrees.</b> It counts this as right (fillers and spelling are normalised away).</div>`;
  }
  function renderCard() {
    const s = currentSpans.find((x) => x.key === state.sel);
    const scored = currentSpans.filter((x) => x.known && state.decisions.has(x.key));
    const hits = scored.filter((x) => x.wrong && state.decisions.get(x.key) === "fix").length;
    const fa = scored.filter((x) => !x.wrong).length;
    const miss = scored.filter((x) => x.wrong && state.decisions.get(x.key) === "ok").length;
    const anyKnown = currentSpans.some((x) => x.known);
    const tally = anyKnown
      ? `<div class="tally"><div><b class="mono">${hits}</b><span>real errors fixed</span></div><div><b class="mono">${fa}</b><span>false alarms</span></div><div><b class="mono">${miss}</b><span>missed</span></div></div>`
      : `<p class="fine">${state.decisions.size} decisions recorded. With no human transcript, flags cannot be checked against the truth.</p>`;
    if (!s) { card.innerHTML = `<h3>No flags at this setting</h3><p class="truth">Lower the threshold or turn on number and negation checks.</p>${tally}`; return; }
    const t = S.units[s.ui], idx = currentSpans.indexOf(s);
    let sent, why, extra = "";
    if (s.type === "word") {
      sent = t.wi.map((wi) => (wi === s.from ? "<mark>" : "") + esc(W[wi].text) + (wi === s.to ? "</mark>" : "")).join(" ");
      const r = [];
      for (let i = s.from; i <= s.to; i++) {
        const w = W[i];
        r.push(w.crit ? `“${esc(w.text)}” is a number or negation` : `“${esc(w.text)}” expected weighted error ${w.v3.toFixed(2)}`);
      }
      why = r.join(" · ");
    } else {
      const words = t.wi.map((wi) => esc(W[wi].text));
      const pos = s.after + 1;
      sent = [...words.slice(0, pos), `<span class="caret">‸</span>`, ...words.slice(pos)].join(" ");
      why = `Gap score ${s.g.score.toFixed(2)} (expected weighted words missing)`;
      extra = `<p class="suggest">${s.g.txt ? `Possibly missing: <b>“${esc(s.g.txt)}”</b> <span class="fine">(what the second model wrote here)</span>` : "Speech may have been skipped here."}</p>`;
    }
    const kind = s.type === "gap" ? "possible missing words" : (s.c ? "number or negation" : "suspect word");
    card.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px">
        <span class="label">Flag ${idx + 1} of ${currentSpans.length} · ${kind} · ${esc(t.role)} · ${fmt(t.s)}</span>
        <span class="fine mono">${(s.c1 - s.c0).toFixed(1)} s clip</span>
      </div>
      <p class="quote ${s.c ? "c" : ""}">${sent}</p>
      ${extra}
      <p class="fine">${why}</p>
      <div class="btns">
        <button class="btn primary" type="button" id="b-play" ${hasAudio ? "" : "disabled"}><svg viewBox="0 0 24 24"><path d="M7 4.5v15l12.5-7.5z" fill="currentColor"/></svg>Play clip <kbd>P</kbd></button>
      </div>
      ${hasAudio ? "" : `<p class="fine">No audio for this consultation. Run <code>asrshift download</code> to enable playback.</p>`}
      ${t.ref !== null ? `<p class="truth">${esc(t.ref)}</p>` : `<p class="truth">No human transcript for this audio.</p>`}
      <div class="btns">
        <button class="btn" type="button" id="b-ok">${s.type === "gap" ? "Nothing missing" : "Correct as written"} <kbd>A</kbd></button>
        <button class="btn" type="button" id="b-fix">Needs a fix <kbd>F</kbd></button>
      </div>
      ${verdictHTML(s, state.decisions.get(s.key))}
      <div class="btns">
        <button class="btn" type="button" id="b-prev">← Previous</button>
        <button class="btn" type="button" id="b-next">Next flag →</button>
      </div>
      <hr>${tally}
      ${t.ref !== null ? `<p class="fine">The grey line is the human transcript of the whole turn, standing in for what you would hear.</p>` : ""}`;
    $("#b-play").onclick = playClip;
    $("#b-ok").onclick = () => decide("ok");
    $("#b-fix").onclick = () => decide("fix");
    $("#b-prev").onclick = () => step(-1);
    $("#b-next").onclick = () => step(1);
  }
  function decide(d) {
    if (!state.sel) return;
    state.decisions.set(state.sel, d);
    const b = doc.querySelector(`.span[data-key="${state.sel}"]`);
    if (b) { b.classList.remove("done-ok", "done-fix"); b.classList.add(`done-${d}`); }
    updateProgress(); renderCard();
  }
  function step(dir) {
    const i = currentSpans.findIndex((x) => x.key === state.sel);
    const n = currentSpans[Math.min(currentSpans.length - 1, Math.max(0, i + dir))];
    if (n) select(n.key, false);
  }
  $("#th").addEventListener("input", (e) => {
    state.th = +e.target.value / 100;
    $("#th-v").textContent = state.th.toFixed(2); $("#s-th").textContent = state.th.toFixed(2);
    buildDoc();
  });
  $("#crit-on").addEventListener("change", (e) => { state.crit = e.target.checked; buildDoc(); });
  document.addEventListener("keydown", (e) => {
    if (state.view !== "review" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.matches("input, textarea, select")) return;
    const k = e.key.toLowerCase();
    if (k === "p") { e.preventDefault(); playClip(); }
    else if (k === "a") decide("ok");
    else if (k === "f") decide("fix");
    else if (k === "j" || e.key === "ArrowDown") { e.preventDefault(); step(1); }
    else if (k === "k" || e.key === "ArrowUp") { e.preventDefault(); step(-1); }
  });

  // ---- compare view ----------------------------------------------------------------------------
  const SERIES = [
    ["window_review", "l-win", "--ink-3", "Review whole windows, least confident first (today)"],
    ["whisper_conf", "l-word", "--flag", "Play words Whisper is least sure of"],
    ["perfect", "l-orc", "--patient", "A perfect verifier (the ceiling)"],
    ["hybrid", "l-hyb", "--good", "Hybrid: verifier, or the whole window when that is cheaper"],
    ["verifier", "l-v3", "--accent", "Verifier: suspect words and gaps"],
  ];
  const chart = $("#chart");
  function drawChart() {
    const C = state.curves;
    if (!C || state.view !== "compare") return;
    const c = C[state.dom];
    if (!c) { chart.innerHTML = ""; $("#takeaway").textContent = "No curves for this unit yet."; return; }
    // the API puts x_seconds at the top level (shared by both units); older fixtures nest it per unit
    const E = state.err, xs = C.x_seconds || c.x_seconds, maxB = xs[xs.length - 1];
    const curves = c[E] || {};
    const present = SERIES.filter(([k]) => Array.isArray(curves[k]));
    const L = 52, R = 620, T = 16, Bt = 340;
    const X = (s) => L + (R - L) * s / maxB, Y = (v) => T + (Bt - T) * (1 - v);
    const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(Math.max(0, v)).toFixed(1)}`).join("");
    let g = `<g class="grid">`;
    for (const v of [0, .25, .5, .75, 1]) g += `<line x1="${L}" x2="${R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
    const step = maxB > 240 ? 60 : 30;
    for (let s = 0; s <= maxB; s += step) g += `<text x="${X(s)}" y="${Bt + 18}" text-anchor="middle">${s} s</text>`;
    g += `</g>`;
    const slider = $("#budget");
    slider.max = String(maxB);
    const bi = xs.reduce((best, x, i) => (Math.abs(x - state.budget) < Math.abs(xs[best] - state.budget) ? i : best), 0);
    const bx = xs[bi];
    const meta = (C.meta && (C.meta[state.dom] || C.meta)) || {};
    const mins = meta.mean_audio_s ? Math.round(meta.mean_audio_s / 60) : null;
    chart.innerHTML = g +
      present.map(([k, cl]) => `<path class="line ${cl}" d="${path(curves[k])}"/>`).join("") +
      `<line class="cursor" x1="${X(bx)}" x2="${X(bx)}" y1="${T}" y2="${Bt}"/>` +
      present.map(([k, , col]) => `<circle cx="${X(bx)}" cy="${Y(Math.max(0, curves[k][bi]))}" r="5" style="fill:var(${col})"/>`).join("") +
      `<text x="${(L + R) / 2}" y="${Bt + 36}" text-anchor="middle">seconds a human listens per consultation${mins ? ` (of ~${mins} min)` : ""}</text>` +
      `<text transform="translate(12 ${(T + Bt) / 2}) rotate(-90)" text-anchor="middle">${E === "critical" ? "critical errors left" : "weighted errors left"}</text>`;
    const pc = (v) => `${Math.round(Math.max(0, v) * 100)}%`;
    $("#r-rows").innerHTML = [...present].reverse().map(([k, , col, lab]) =>
      `<div class="ro"><i style="background:var(${col})"></i><span>${lab}</span><b class="mono">${pc(curves[k][bi])}</b></div>`).join("");
    $("#r-title").textContent = E === "critical" ? "Number, dose and negation errors left" : "Weighted errors left in the transcript";
    $("#budget-v").textContent = bx;
    const nf = new Intl.NumberFormat("en");
    const unitName = state.dom === "window" ? "30-s windows" : "speaker turns";
    $("#cmp-n").innerHTML = [
      meta.consultations ? `${meta.consultations} PriMock57 test consultations, ${unitName}` : unitName,
      meta.weighted_errors ? `${nf.format(Math.round(meta.weighted_errors))} weighted errors` : "",
      meta.critical_errors ? `${nf.format(meta.critical_errors)} of them on numbers, doses or negations` : "",
    ].filter(Boolean).join(" · ") + "." + (C.meta?.note ? ` ${esc(C.meta.note)}` : "");
    // takeaway from the data itself, not hard-coded claims
    const v = curves.verifier, w = curves.window_review, h = curves.hybrid;
    if (v && w) {
      const i60 = xs.indexOf(60) >= 0 ? xs.indexOf(60) : bi;
      const catchUp = xs.findIndex((x, i) => x > 0 && w[i] <= v[i]);
      let txt = `<b>At ${xs[i60]} s of listening per consultation</b>, the verifier leaves ${pc(v[i60])} of the ${E === "critical" ? "critical" : "weighted"} errors, against ${pc(w[i60])} when whole ${state.dom === "window" ? "windows" : "turns"} are reviewed. `;
      txt += catchUp > 0 ? `Whole-${state.dom} review catches up at about ${xs[catchUp]} s, because errors bunch up in bad ${state.dom}s. `
                         : `It stays ahead at every budget up to ${maxB} s. `;
      if (h) {
        const best = h[i60] < v[i60] ? "the hybrid" : "the plain verifier";
        txt += `The hybrid, which reviews the whole ${state.dom} when its flags would cost more, leaves ${pc(h[i60])} at ${xs[i60]} s, so ${best} is ahead there. `;
      }
      if (curves.whisper_conf) txt += `Playing the words Whisper itself is least sure of leaves ${pc(curves.whisper_conf[i60])}.`;
      $("#takeaway").innerHTML = txt;
    } else $("#takeaway").textContent = "";
  }
  $$(".seg button[data-dom]").forEach((b) => b.addEventListener("click", () => {
    state.dom = b.dataset.dom;
    $$(".seg button[data-dom]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    drawChart();
  }));
  $$(".seg button[data-err]").forEach((b) => b.addEventListener("click", () => {
    state.err = b.dataset.err;
    $$(".seg button[data-err]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    drawChart();
  }));
  $("#budget").addEventListener("input", (e) => { state.budget = +e.target.value; drawChart(); });

  async function ensureCurves() {
    if (state.curves) return drawChart();
    $("#takeaway").textContent = "Loading curves…";
    try { state.curves = await API.curves(); drawChart(); }
    catch (err) {
      chart.innerHTML = ""; $("#r-rows").innerHTML = "";
      $("#takeaway").innerHTML = `Could not load the review curves (${esc(err.message)}). Train the verifier with <code>asrshift verifier</code> to create them.`;
    }
  }

  // ---- consultations ---------------------------------------------------------------------------
  let consultations = [], lastConsult = null;
  function showSplit() {
    const b = $("#split-badge");
    if (S.kind === "upload") { b.hidden = false; b.className = "badge upload"; b.textContent = "uploaded"; return; }
    b.hidden = !S.split;
    b.className = `badge ${S.split === "test" ? "test" : "train"}`;
    b.textContent = S.split === "test" ? "test · never seen in training" : S.split ? "used in training" : "";
  }
  function setFine() {
    const fine = $("#live-fine");
    if (S.kind === "upload") fine.textContent = "Whisper-small transcribing your file now. There is no human transcript, so flags cannot be checked against the truth.";
    else if (S.kind === "live") fine.textContent = "Whisper-small decoding each turn from its own channel right now; the verifier scores it as it arrives. Word timing inside a turn is spread evenly.";
    else fine.textContent = "Whisper-small's saved output on this audio, replayed at its timings. Word timing inside each turn is spread evenly.";
    $("#ref-wrap").hidden = S.kind === "upload";
    $("#empty-text").innerHTML = S.kind === "live"
      ? "Press <b>Start transcribing</b>. Whisper decodes each turn from the audio, and the verifier flags words that may be wrong and marks ‸ where words may be missing."
      : S.kind === "upload" ? "Waiting for Whisper's first segment…"
      : "Press play. The transcript appears as the doctor and patient speak. The verifier lights up words that may be wrong and marks ‸ where words may be missing.";
  }
  async function openConsultation(id, opts = {}) {
    if (!id || id === "__upload__") return;
    lastConsult = id;
    clock.pause();
    const info = consultations.find((c) => c.id === id) || { id };
    const sel = $("#consult");
    sel.querySelector('option[value="__upload__"]')?.remove();
    sel.value = id;
    notice(`Loading <b>${esc(id)}</b>…`);
    let payload;
    try { payload = await API.payload(id); }
    catch (err) {
      newSession("none", info); rebuildView();
      notice(`Could not load ${esc(id)}: ${esc(err.message)}.`, true);
      return;
    }
    mockPayload = payload;
    const keep = opts.keepAudio && hasAudio && S.id === id;
    newSession("saved", { id, split: payload.split ?? info.split, duration: payload.duration || info.duration });
    payload.units.forEach(addUnit);
    $("#meta").innerHTML = `PriMock57 · ${esc(id)}<br>${fmt(S.duration)} · ${payload.units.length} turns · whisper-small`;
    showSplit();
    if (!keep) {
      audioNote("Loading audio…");
      const ok = info.has_audio === false ? false : await loadAudio(API.audioUrl(id));
      if (S.id !== id) return;  // another consultation was picked meanwhile
      audioNote(ok ? "" : "No audio for this consultation yet. Run <code>asrshift download</code> to enable playback. Play still steps through the transcript.");
    }
    playBtn.disabled = false;
    notice("");
    setMode(opts.mode || "replay");
    if (state.mode === "run") startRunSession(false);
    setFine();
    rebuildView();
    tick(true);
  }
  $("#consult").addEventListener("change", (e) => { setMode("replay"); openConsultation(e.target.value); });

  function rebuildView() {
    currentSpans = spans();
    if (state.view === "live") { buildLive(); tick(true); }
    if (state.view === "review") buildDoc();
    if (state.view === "compare") ensureCurves();
    drawWave();
  }

  // ---- routing ---------------------------------------------------------------------------------
  function route() {
    const v = ["live", "review", "compare"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "live";
    if (v !== state.view && !(live.active && !live.done)) { clock.pause(); clipEnd = null; }
    state.view = v;
    $$(".tabs a").forEach((a) => (a.dataset.view === v ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
    $("#view-live").hidden = v !== "live";
    $("#view-review").hidden = v !== "review";
    $("#view-compare").hidden = v !== "compare";
    $(".player").hidden = v === "compare";
    rebuildView();
  }
  window.addEventListener("hashchange", route);

  async function init() {
    route();
    notice(MOCK ? "Mock mode: sample data from <code>mock/</code>, no server needed." : "Loading consultations…");
    try { consultations = await API.consultations(); }
    catch (err) {
      notice(`Could not reach the Second Listen server (${esc(err.message)}). Start it from the repository root with ` +
             `<code>uvicorn app.server:app --port 8765</code>, or open <a href="?mock=1">?mock=1</a> to try the interface with sample data.`, true);
      $("#consult").innerHTML = "<option>No consultations</option>";
      return;
    }
    const sel = $("#consult");
    if (!consultations.length) { notice("The server has no consultations. Run <code>asrshift prepare</code> and train the verifier first.", true); return; }
    const order = { test: 0, recalibration: 1 };
    consultations.sort((a, b) => (order[a.split] ?? 2) - (order[b.split] ?? 2) || (b.has_audio - a.has_audio) || a.id.localeCompare(b.id));
    sel.innerHTML = consultations.map((c) =>
      `<option value="${esc(c.id)}">${esc(c.id)} · ${c.split === "test" ? "test" : "used in training"}${c.duration ? ` · ${fmt(c.duration)}` : ""}${c.has_audio ? "" : " · no audio"}</option>`).join("");
    sel.disabled = false;
    if (MOCK) notice("");
    const first = consultations.find((c) => c.has_audio && c.split === "test") || consultations[0];
    await openConsultation(first.id);
    if (MOCK) notice("Mock mode: sample data from <code>mock/</code>, no server needed. Run Whisper now replays the saved turns as if decoding at 2.5× real time; uploads need the server.");
  }
  init();
})();
