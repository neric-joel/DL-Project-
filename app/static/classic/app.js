"use strict";

const state = { eval: "eka_test", policy: "P3", point: "budget_0.10", show: "all", selected: null, meta: null, seq: 0, detailSeq: 0, live: false };
const DOTS = { eka_test: "var(--eka)", pm_turn_test: "var(--turn)", pm_window_test: "var(--win)" };
const FILTERS = {
  all: "All",
  review: "Sent to review",
  accepted_errors: "Accepted but wrong",
  accepted_critical: "Accepted with dose/negation error",
};
const ICON_REVIEW = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="2.4"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>';
const ICON_ACCEPT = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const ICON_WARN = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l10 18H2z" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/><path d="M12 10v5M12 18v.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>';

const $ = (s, el = document) => el.querySelector(s);
// whole percents, but one decimal below 10% so that 0.3% does not read as 0%
const pct = (x, d) => (x == null || Number.isNaN(x) ? "—" : `${(100 * x).toFixed(d ?? (Math.abs(x) < 0.1 && x !== 0 ? 1 : 0))}%`);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function pill(decision) {
  return decision === "review"
    ? `<span class="pill review">${ICON_REVIEW}Review</span>`
    : `<span class="pill accept">${ICON_ACCEPT}Accept</span>`;
}

function available(ev, pol, pt) {
  return state.meta.available[`${ev}|${pol}|${pt}`] !== undefined;
}

function buildSeg(id, options, key, decorate) {
  const root = $(`#${id}`);
  root.innerHTML = "";
  for (const [value, label] of Object.entries(options)) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.dataset.value = value;
    b.innerHTML = decorate ? decorate(value, label) : esc(label);
    b.addEventListener("click", () => { state[key] = value; fixSelection(key); refresh().catch(showError); });
    root.appendChild(b);
  }
}

function syncSegs() {
  const sets = [["seg-eval", "eval"], ["seg-policy", "policy"], ["seg-point", "point"], ["chips", "show"]];
  for (const [id, key] of sets) {
    for (const b of $(`#${id}`).children) {
      b.setAttribute("aria-checked", String(b.dataset.value === state[key]));
      if (key === "policy") b.disabled = !available(state.eval, b.dataset.value, state.point) && !Object.keys(state.meta.points).some((p) => available(state.eval, b.dataset.value, p));
      if (key === "point") b.disabled = !available(state.eval, state.policy, b.dataset.value);
    }
  }
}

// Keep the control the user just changed; adjust the other one if the combination does not exist
// (e.g. P4 is not defined on Eka, and the plug-in rule needs probabilities, which P1 does not give).
function fixSelection(changed) {
  if (available(state.eval, state.policy, state.point)) return;
  const points = Object.keys(state.meta.points);
  const policies = Object.keys(state.meta.policies);
  if (changed !== "point") {
    const pt = points.find((p) => available(state.eval, state.policy, p));
    if (pt) { state.point = pt; return; }
  }
  state.policy = policies.find((p) => available(state.eval, p, state.point)) || "P3";
  if (!available(state.eval, state.policy, state.point)) state.point = "budget_0.10";
}

function kpi(id, value, sub, alert = false) {
  const el = $(`#${id}`);
  $("[data-v]", el).textContent = value;
  $("[data-sub]", el).innerHTML = sub;
  el.classList.toggle("alert", alert);
}

function showError(e) {
  $("#note").innerHTML = `<span class="flag" role="alert">${ICON_WARN} Could not load this view: ${esc(e.message)}</span>`;
}

async function refresh() {
  syncSegs();
  const seq = ++state.seq;
  const q = new URLSearchParams({ eval_set: state.eval, policy: state.policy, point: state.point });
  const [s, queue] = await Promise.all([api(`/api/summary?${q}`), api(`/api/queue?${q}&show=${encodeURIComponent(state.show)}`)]);
  if (seq !== state.seq) return;  // a newer selection is already on its way

  $("#note").textContent = s.feasible ? s.note : "No threshold meets this target on the tuning data, so every transcript goes to review.";
  const drift = s.promised_review != null && Math.abs(s.review_rate - s.promised_review) > 0.03;
  kpi("kpi-review", pct(s.review_rate),
    s.promised_review != null
      ? `${drift ? `<span class="flag">${ICON_WARN}</span>` : ""}promised <b>${pct(s.promised_review)}</b> · 95% CI ${pct(s.review_rate_ci[0])}–${pct(s.review_rate_ci[1])}`
      : `of ${s.units.toLocaleString()} transcripts`, drift);
  kpi("kpi-caught", pct(s.err_caught), `${pct(s.base_rate)} of all transcripts have WER &gt; 10%`);
  const over = s.promised_accepted_err != null && s.accepted_err_rate > s.promised_accepted_err + 0.02;
  kpi("kpi-acc", pct(s.accepted_err_rate),
    s.promised_accepted_err != null
      ? `${over ? `<span class="flag">${ICON_WARN}</span>` : ""}target <b>≤ ${pct(s.promised_accepted_err)}</b> · 95% CI ${pct(s.accepted_err_ci[0])}–${pct(s.accepted_err_ci[1])}`
      : `95% CI ${pct(s.accepted_err_ci[0])}–${pct(s.accepted_err_ci[1])}`, over);
  kpi("kpi-crit", pct(s.accepted_crit_rate), "missed or wrong number or negation");

  const tbody = $("#rows");
  tbody.innerHTML = queue.rows.map((r) => `
    <tr data-id="${esc(r.unit_id)}" aria-selected="${r.unit_id === state.selected}" tabindex="0">
      <td><div class="riskbar"><span style="width:${Math.max(2, Math.min(1, r.risk) * 44)}px"></span>${r.risk.toFixed(2)}</div></td>
      <td>${pill(r.decision)}</td>
      <td class="${r.err ? "wrong" : ""}">${pct(r.wer)}</td>
      <td>${r.conf.toFixed(2)}</td>
      <td class="txt" title="${esc(r.hypothesis)}">${esc(r.hypothesis) || "<em>(empty)</em>"}</td>
    </tr>`).join("");
  for (const tr of tbody.children) {
    const open = () => select(tr.dataset.id).catch(showError);
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
  }
  $("#queue-foot").textContent = `${queue.total.toLocaleString()} transcripts${queue.total > queue.rows.length ? `, showing ${queue.rows.length}` : ""}.`;
  if (state.selected && !state.live) renderDetail(state.selected);
}

async function select(id) {
  if (state.live) return;  // keep the in-flight live transcription on screen
  state.selected = id;
  for (const tr of $("#rows").children) tr.setAttribute("aria-selected", String(tr.dataset.id === id));
  await renderDetail(id);
}

function clearDetail() {
  state.selected = null;
  $("#detail").innerHTML = `<div class="empty"><p>Select a transcript to hear the audio, see what Whisper got wrong, and how each policy scored it.</p></div>`;
}

function renderDiff(ops) {
  return ops.map((o) => {
    if (o.op === "C") return `<span class="tok">${esc(o.ref)}</span>`;
    if (o.op === "S") return `<span class="tok"><span class="sub-ref">${esc(o.ref)}</span> <span class="sub-tok">${esc(o.hyp)}</span></span>`;
    if (o.op === "D") return `<span class="tok del">${esc(o.ref)}</span>`;
    if (o.op === "I") return `<span class="tok ins">${esc(o.hyp)}</span>`;
    return `<span class="tok wild">[unintelligible${o.hyp ? `: ${esc(o.hyp)}` : ""}]</span>`;
  }).join(" ");
}

function renderWords(words) {
  if (!words.length) return "<em>(no words)</em>";
  return words.map((w) => `<span style="background: rgba(235,104,52,${(0.06 + 0.6 * (1 - w.p)).toFixed(2)})" title="p = ${w.p}">${esc(w.w.trim())}</span>`).join(" ");
}

function renderRisks(risks) {
  const names = state.meta.policies;
  return Object.entries(risks).map(([pol, r]) => {
    const thr = r.threshold == null ? "none (every transcript is reviewed)" : r.threshold.toFixed(2);
    return `
    <div class="risk-row">
      <span>${esc(r.label || names[pol] || pol)}</span>
      <div class="risk-track" role="img" aria-label="risk ${r.risk.toFixed(2)}, threshold ${thr}">
        <div class="risk-fill" style="width:${(Math.min(1, Math.max(0, r.risk)) * 100).toFixed(1)}%"></div>
        ${r.threshold != null && r.threshold <= 1 ? `<div class="risk-thr" style="left:${(Math.max(0, r.threshold) * 100).toFixed(1)}%" title="threshold ${thr}"></div>` : ""}
      </div>
      <span>${pill(r.decision)}</span>
    </div>`;
  }).join("");
}

async function renderDetail(id) {
  const seq = ++state.detailSeq;
  const q = new URLSearchParams({ eval_set: state.eval, point: state.point });
  const u = await api(`/api/unit/${encodeURIComponent(id)}?${q}`);
  if (seq !== state.detailSeq) return;
  if (!u.in_set) { clearDetail(); return; }  // the selected transcript is not in the new evaluation set
  const root = $("#detail");
  root.innerHTML = "";
  root.appendChild($("#tpl-detail").content.cloneNode(true));
  const kind = u.dataset === "eka" ? `Eka · ${u.subset.replace("_", " ")}` : `PriMock57 · ${u.subset.replace("_", " ")}`;
  $("[data-kind]", root).textContent = `${kind} · ${u.duration.toFixed(1)} s`;
  $("[data-id]", root).textContent = u.unit_id;
  const mine = u.risks[state.policy];
  $("[data-decision]", root).innerHTML = mine ? pill(mine.decision) : "";
  const audio = $("[data-audio]", root);
  if (u.audio) audio.src = `/api/audio/${encodeURIComponent(id)}`; else audio.replaceWith(Object.assign(document.createElement("p"), { className: "facts", textContent: "Audio needs the downloaded datasets." }));
  $("[data-risks]", root).innerHTML = renderRisks(u.risks);
  $("[data-words]", root).innerHTML = renderWords(u.words);
  $("[data-diff]", root).innerHTML = renderDiff(u.ops);
  const e = u.errors;
  $("[data-facts]", root).innerHTML = `WER <b>${pct(u.wer, 1)}</b> (${e.sub} substituted, ${e.del} deleted, ${e.ins} inserted of ${e.n_ref} words) · ` +
    `sequence confidence <b>${u.conf.toFixed(2)}</b> · dose/negation tokens: ${u.critical.missed} missed, ${u.critical.false} wrong`;
  const btn = $("[data-live]", root);
  btn.disabled = !u.audio || !state.meta.live;
  btn.addEventListener("click", () => live(id, root));
}

async function live(id, root) {
  const btn = $("[data-live]", root);
  const status = $("[data-live-status]", root);
  const out = $("[data-live-out]", root);
  btn.disabled = true;
  state.live = true;
  status.textContent = "Transcribing on the GPU…";
  try {
    const r = await api(`/api/transcribe/${encodeURIComponent(id)}`, { method: "POST" });
    const point = "budget_0.10";  // the exported policies carry budget and risk-target thresholds
    const rows = Object.entries(r.risk).filter(([, v]) => v != null).map(([pol, v]) => {
      const d = r.decisions[pol] && r.decisions[pol][point];
      return `${esc(state.meta.policies[pol] || pol)}: risk <b>${v.toFixed(2)}</b> ${d ? pill(d) : ""}`;
    }).join("<br>");
    out.innerHTML = `<p><b>Fresh transcript</b> (${r.seconds}s for ${r.audio_s.toFixed(1)}s of audio):</p>` +
      `<p>${renderWords(r.words)}</p><p>${rows}</p><p class="facts">Decisions shown at the 10% review budget.</p>`;
    out.hidden = false;
    status.textContent = "Done.";
  } catch (e) {
    status.textContent = `Could not run Whisper: ${e.message}`;
  } finally {
    btn.disabled = false;
    state.live = false;
  }
}

async function init() {
  state.meta = await api("/api/meta");
  buildSeg("seg-eval", state.meta.eval_sets, "eval", (v, l) => `<span class="dot" style="background:${DOTS[v]}"></span>${esc(l)}`);
  buildSeg("seg-policy", state.meta.policies, "policy");
  buildSeg("seg-point", state.meta.points, "point");
  buildSeg("chips", FILTERS, "show");
  $("#status").textContent = state.meta.live ? "Live decoding available · whisper-small" : "Results only (datasets not downloaded)";
  const params = new URLSearchParams(location.search);
  for (const k of ["eval", "policy", "point", "show"]) if (params.get(k)) state[k] = params.get(k);
  fixSelection();
  await refresh();
}

init().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<p role="alert">Could not load results: ${esc(e.message)}</p>`); });
