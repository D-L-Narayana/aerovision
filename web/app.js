import { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";
env.allowLocalModels = false;
const $ = (s) => document.querySelector(s);
const MODEL = "Xenova/clip-vit-base-patch32";
let processor, vision, TXT, PROBE, RESULTS, lastScene = null, lastThermal = null;

// ---------- CLIP heads ----------
const softmax = (z) => { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
async function embedImage(src) {
  const img = await RawImage.read(src); const inputs = await processor(img);
  const t0 = performance.now(); const { image_embeds } = await vision(inputs); const ms = performance.now() - t0;
  const v = Array.from(image_embeds.data); const n = Math.hypot(...v); return { v: v.map((x) => x / n), ms };
}
function heads(v) {
  const zs = softmax(TXT.T.map((t) => 100 * dot(v, t)));
  const pr = softmax(PROBE.W.map((w, i) => dot(v, w) + PROBE.b[i]));
  return { zs, pr };
}
const SENS = { airplane: "restricted airspace likely — airport", runway: "restricted airspace likely — airport", denseresidential: "flight over people risk — require waiver", mediumresidential: "flight over people risk — require waiver", mobilehomepark: "flight over people risk — require waiver", storagetanks: "industrial site — check exclusion zones", harbor: "crane / mast exclusion zones", freeway: "moving vehicles — no hover", overpass: "moving vehicles — no hover", beach: "crowds possible — check occupancy" };
function bars(el, p, cls) {
  const idx = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 5);
  el.innerHTML = idx.map(([v, i]) => `<div class="bar ${cls}"><span>${TXT.classes[i]}</span><span class="t"><i style="width:${(v * 100).toFixed(1)}%"></i></span><span class="n">${(v * 100).toFixed(1)}%</span></div>`).join("");
}
async function classify(src) {
  $("#status").textContent = "Encoding image…";
  const img = new Image(); img.src = src; await img.decode();
  const ctx = $("#preview").getContext("2d"); ctx.drawImage(img, 0, 0, 224, 224);
  const { v, ms } = await embedImage(src); const { zs, pr } = heads(v);
  bars($("#zs"), zs, "zs"); bars($("#pr"), pr, "pr"); $("#zs-lat").textContent = `${ms.toFixed(0)} ms`;
  const top = pr.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 2);
  const sens = [...new Set(top.filter(([p, i]) => p >= 0.3 && SENS[TXT.classes[i]]).map(([, i]) => SENS[TXT.classes[i]]))];
  $("#sens").textContent = sens.length ? "⚠ " + sens.join(" · ") : "";
  lastScene = { scene: top.map(([p, i]) => ({ label: TXT.classes[i], probability: +p.toFixed(4), source: "linear_probe" })), zero_shot_top: TXT.classes[zs.indexOf(Math.max(...zs))], flight_sensitivity: sens, encode_ms: +ms.toFixed(0) };
  $("#status").textContent = `Ready · ${MODEL} in-browser (WASM) · last image encoded in ${ms.toFixed(0)} ms`;
  report();
}

// ---------- thermal (port of aerovision/thermal.py) ----------
const H = 96, W = 128, NS = 8;
function randn() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function makeFrame() {
  const f = new Float32Array(H * W); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) f[y * W + x] = 38 + randn() * 1.2 + (x / (W - 1)) * 3 - 1.5;
  const n = Math.floor(Math.random() * 4); const deltas = [6, 8, 12, 15, 18, 22, 28, 35];
  for (let k = 0; k < n; k++) { const d = deltas[Math.floor(Math.random() * deltas.length)], w = 4 + Math.floor(Math.random() * 10), h = 4 + Math.floor(Math.random() * 10), x0 = Math.floor(Math.random() * (W - w)), y0 = Math.floor(Math.random() * (H - h));
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) f[(y0 + y) * W + x0 + x] += d * Math.exp(-(((y - h / 2) / (h / 2.2)) ** 2 + ((x - w / 2) / (w / 2.2)) ** 2)); }
  if (Math.random() < 0.3) { const s = Math.floor(Math.random() * NS), d = [6, 9, 12][Math.floor(Math.random() * 3)]; const b = bandRows(s); for (let y = b[0]; y < b[1]; y++) for (let x = 0; x < W; x++) f[y * W + x] += d; }
  return f;
}
const bandRows = (s) => { const base = Math.floor(H / NS), extra = H % NS; const start = s * base + Math.min(s, extra); return [start, start + base + (s < extra ? 1 : 0)]; };
const median = (a) => { const b = Float32Array.from(a).sort(); return b.length % 2 ? b[(b.length - 1) / 2] : (b[b.length / 2 - 1] + b[b.length / 2]) / 2; };
function detect(f) {
  const med = median(f); const ref = new Float32Array(H * W); const strings = [];
  for (let s = 0; s < NS; s++) { const [a, b] = bandRows(s); const m = median(f.subarray(a * W, b * W)); ref.fill(m, a * W, b * W); const d = m - med; if (d >= 5) strings.push({ anomaly_type: "hot_string", string: s, delta_c: +d.toFixed(1), severity: d >= 10 ? "high" : "medium" }); }
  const ex = f.map((v, i) => v - ref[i]); const seen = new Uint8Array(H * W); const spots = [];
  for (let i = 0; i < H * W; i++) { if (ex[i] < 10 || seen[i]) continue; const st = [i]; seen[i] = 1; let n = 0, minx = W, maxx = 0, miny = H, maxy = 0, mx = 0;
    while (st.length) { const j = st.pop(); n++; const y = (j / W) | 0, x = j % W; minx = Math.min(minx, x); maxx = Math.max(maxx, x); miny = Math.min(miny, y); maxy = Math.max(maxy, y); mx = Math.max(mx, ex[j]);
      for (const [dy, dx] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const ny = y + dy, nx = x + dx; if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue; const k = ny * W + nx; if (ex[k] >= 10 && !seen[k]) { seen[k] = 1; st.push(k); } } }
    if (n >= 4) spots.push({ anomaly_type: "hotspot", bbox: [minx, miny, maxx - minx + 1, maxy - miny + 1], delta_c: +mx.toFixed(1), severity: mx >= 20 ? "high" : "medium" }); }
  spots.sort((a, b) => b.delta_c - a.delta_c); return { anomalies: [...spots, ...strings], median: med };
}
function drawThermal(f, det) {
  const c = $("#thermal"), ctx = c.getContext("2d"), img = ctx.createImageData(W, H);
  for (let i = 0; i < H * W; i++) { const v = Math.min(1, Math.max(0, (f[i] - 30) / 40)); img.data.set([255 * Math.sqrt(v), 255 * v * v, 255 * (1 - v), 255], i * 4); }
  const off = new OffscreenCanvas(W, H); off.getContext("2d").putImageData(img, 0, 0); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, c.width, c.height);
  const sx = c.width / W, sy = c.height / H; ctx.lineWidth = 2;
  for (const a of det.anomalies) { ctx.strokeStyle = a.severity === "high" ? "#ff6b6b" : "#f6ad55"; if (a.bbox) ctx.strokeRect(a.bbox[0] * sx - 2, a.bbox[1] * sy - 2, a.bbox[2] * sx + 4, a.bbox[3] * sy + 4); else { const [r0, r1] = bandRows(a.string); ctx.setLineDash([6, 4]); ctx.strokeRect(1, r0 * sy, c.width - 2, (r1 - r0) * sy); ctx.setLineDash([]); } }
  $("#spots").innerHTML = (det.anomalies.map((a) => `<div class="${a.severity}">${a.anomaly_type}${a.bbox ? ` @ (${a.bbox[0]},${a.bbox[1]}) ${a.bbox[2]}×${a.bbox[3]}px` : ` S-${a.string}`} · Δ ${a.delta_c} °C · ${a.severity}</div>`).join("") || `<div class="muted">no anomaly ≥ 10 °C</div>`) + `<div class="muted">frame median ${det.median.toFixed(1)} °C</div>`;
}
function newFrame() { const f = makeFrame(); const t0 = performance.now(); const det = detect(f); det.ms = performance.now() - t0; drawThermal(f, det); lastThermal = det; report(); }

// ---------- report (mirrors aerovision/report.py) ----------
function report() {
  const rank = { none: 0, low: 1, medium: 2, high: 3 }; const an = lastThermal?.anomalies || [];
  const maxSev = an.reduce((m, a) => (rank[a.severity] > rank[m] ? a.severity : m), "none");
  const action = { none: "none", low: "log", medium: "schedule_within_7_days", high: "same_day_work_order" }[maxSev];
  const conf = an.length === 0 || Math.max(0, ...an.map((a) => a.delta_c)) >= 10 ? 0.95 : 0.5;
  const r = { frame_id: Math.random().toString(16).slice(2, 10), scene: lastScene?.scene || [], flight_sensitivity: lastScene?.flight_sensitivity || [], anomalies: an, max_severity: maxSev, recommended_action: action, confidence: conf, verified: conf >= 0.6, narrative: `Deterministic detector found ${an.length} anomalies in ${lastThermal ? lastThermal.ms.toFixed(2) : "–"} ms; scene top-1 (probe) ${lastScene?.scene?.[0]?.label || "–"}${lastScene ? `, zero-shot ${lastScene.zero_shot_top}` : ""}.` };
  $("#report").textContent = JSON.stringify(r, null, 2);
}

// ---------- benchmark widgets ----------
function renderBench(r) {
  const z = r.zero_shot, p = r.linear_probe;
  $("#bench").innerHTML = `<tr><th>method (UC Merced, 21 classes, 2100 imgs)</th><th>top-1</th></tr><tr><td>zero-shot, single prompt</td><td>${(z.top1_single_prompt_all * 100).toFixed(1)}%</td></tr><tr><td>zero-shot, 4-template ensemble</td><td>${(z.top1_all * 100).toFixed(1)}% <span class="muted">(top-3 ${(z.top3_all * 100).toFixed(1)}%)</span></td></tr><tr><td>linear probe, 50 img/class</td><td class="best">${(p.top1_test * 100).toFixed(1)}%</td></tr><tr><td>thermal detector (200 frames, ≥10 °C)</td><td>P ${(r.thermal_detector.precision * 100).toFixed(0)}% · R ${(r.thermal_detector.recall * 100).toFixed(1)}%</td></tr>`;
  $("#bench-foot").textContent = `${r.model} · PyTorch ${r.device} ${r.clip_latency_ms_per_image_cpu} ms/image · severity agreement ${(r.thermal_detector.severity_agreement * 100).toFixed(1)}% · ${r.split}`;
  const c = $("#curve"), ctx = c.getContext("2d"), ks = Object.keys(p.few_shot_curve), vs = Object.values(p.few_shot_curve); const L = 36, R = 10, T = 14, B = 28, w = c.width - L - R, h = c.height - T - B;
  ctx.clearRect(0, 0, c.width, c.height); ctx.strokeStyle = "#1f2a36"; ctx.fillStyle = "#8b98a8"; ctx.font = "11px ui-monospace, monospace";
  for (let i = 0; i <= 4; i++) { const yy = T + h - (h * i) / 4; ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(L + w, yy); ctx.stroke(); ctx.fillText(`${60 + i * 10}%`, 2, yy + 4); }
  const X = (i) => L + (w * i) / (ks.length - 1), Y = (v) => T + h - h * ((v - 0.6) / 0.4);
  ctx.setLineDash([5, 4]); ctx.strokeStyle = "#8b98a8"; ctx.beginPath(); ctx.moveTo(L, Y(z.top1_test)); ctx.lineTo(L + w, Y(z.top1_test)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = "#f6ad55"; ctx.lineWidth = 2; ctx.beginPath(); vs.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke();
  ctx.fillStyle = "#f6ad55"; vs.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(i), Y(v), 3.5, 0, 7); ctx.fill(); }); ctx.fillStyle = "#8b98a8"; ks.forEach((k, i) => ctx.fillText(k, X(i) - 6, c.height - 10));
}

// ---------- wiring ----------
$("#regen").addEventListener("click", newFrame);
const drop = $("#drop"); drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); }); drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) classify(URL.createObjectURL(f)); });
$("#file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) classify(URL.createObjectURL(f)); });

(async () => {
  [TXT, PROBE, RESULTS] = await Promise.all(["text_embs.json", "probe_head.json", "results.json"].map((u) => fetch(u).then((r) => r.json())));
  renderBench(RESULTS); newFrame();
  const samples = await (await fetch("samples.json")).json();
  $("#samples").innerHTML = samples.map((s) => `<img src="${s.file}" title="${s.label}" alt="${s.label}" loading="lazy" />`).join("");
  $("#samples").addEventListener("click", (e) => { if (e.target.tagName === "IMG") classify(e.target.src); });
  processor = await AutoProcessor.from_pretrained(MODEL);
  vision = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { dtype: "q8", progress_callback: (p) => { if (p.status === "progress") $("#status").textContent = `Loading ${MODEL}: ${p.file} ${Math.round(p.progress || 0)}%`; } });
  $("#status").textContent = `Ready · ${MODEL} running in-browser (WASM, q8) · pick a sample or drop an image`;
  classify(samples[19].file);
})().catch((e) => { $("#status").textContent = "Failed to load: " + e.message; console.error(e); });
