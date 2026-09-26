"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, kids = []) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const kid of [].concat(kids)) if (kid) node.append(kid);
  return node;
};
const num = (v, d = 0) => (v === null || v === undefined ? "-" :
  Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
const usd = (v) => "$" + Number(v || 0).toFixed(5);
const get = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + " -> " + res.status);
  return res.json();
};

const state = { overview: null, clusters: [], view: "runs", run: null };

// --- routing -----------------------------------------------------------------------------------

const VIEWS = ["runs", "run", "clusters", "cost", "validation", "regression"];

function show(view) {
  state.view = view;
  for (const name of VIEWS) {
    const node = $("#view-" + name);
    if (node) node.hidden = name !== view;
  }
  for (const btn of document.querySelectorAll("nav button")) {
    btn.setAttribute("aria-selected", String(btn.dataset.view === view));
  }
}

for (const btn of document.querySelectorAll("nav button")) {
  btn.addEventListener("click", () => {
    const view = btn.dataset.view;
    show(view);
    if (view === "clusters") renderClusters();
    if (view === "cost") renderCost();
    if (view === "validation") renderValidation();
    if (view === "regression") renderRegression();
  });
}
$("#back").addEventListener("click", () => show("runs"));

// --- runs list ---------------------------------------------------------------------------------

function fillSelect(node, label, values) {
  node.replaceChildren(el("option", { value: "", text: label }));
  for (const v of values) node.append(el("option", { value: v, text: v }));
}

async function renderRuns() {
  const params = new URLSearchParams({ limit: "200", order: $("#f-order").value });
  for (const [key, sel] of [["q", "#f-q"], ["cluster", "#f-cluster"], ["cohort", "#f-cohort"],
                            ["pathology", "#f-pathology"]]) {
    const v = $(sel).value;
    if (v) params.set(key, v);
  }
  const rows = await get("/api/runs?" + params);
  $("#runs-count").textContent = rows.length + " shown";
  const body = $("#runs-body");
  body.replaceChildren(...rows.map((r) => {
    const tr = el("tr", { "data-href": r.run_id }, [
      el("td", { class: "mono", text: r.run_id }),
      el("td", { class: r.n_errors ? "bad" : "", text: r.status || "-" }),
      el("td", { class: "n", text: num(r.n_events) }),
      el("td", { class: "n", text: num(r.n_tool_calls) }),
      el("td", { class: "n", text: num(r.tokens_in + r.tokens_out) }),
      el("td", { class: "n", text: usd(r.cost_usd) }),
      el("td", { class: "n", text: num(r.wall_ms) }),
      el("td", { class: "mono muted", text: r.cluster_slug || "-" }),
      el("td", {}, r.pathologies.map((p) => el("span", { class: "pill bad", text: p }))),
    ]);
    tr.addEventListener("click", () => openRun(r.run_id));
    return tr;
  }));
}

for (const sel of ["#f-q", "#f-cluster", "#f-cohort", "#f-pathology", "#f-order"]) {
  $(sel).addEventListener("input", renderRuns);
}

// --- one run -----------------------------------------------------------------------------------

function statRow(pairs) {
  return el("div", { class: "stats" }, pairs.map(([k, v]) =>
    el("div", { class: "stat" }, [el("div", { class: "k", text: k }),
                                 el("div", { class: "v", text: v })])));
}

async function openRun(runId) {
  const run = await get("/api/runs/" + encodeURIComponent(runId));
  state.run = run;
  show("run");
  $("#run-title").textContent = run.run_id;
  $("#run-stats").replaceChildren(statRow([
    ["status", run.status || "-"],
    ["cluster", run.cluster_slug || "-"],
    ["spans", num(run.spans.length)],
    ["wall", num(run.wall_ms) + " ms"],
    ["cost", usd(run.cost_usd)],
    ["tokens", num(run.tokens_in + run.tokens_out)],
    ["errors", num(run.n_errors)],
  ]));
  $("#run-findings").replaceChildren(...(run.findings.length
    ? [el("h3", { text: "Detectors" }),
       ...run.findings.map((f) => el("pre", {
         text: f.pathology + "  strength " + f.strength.toFixed(2) + "\n" +
               JSON.stringify(f.evidence, null, 1),
       }))]
    : [el("h3", { text: "Detectors: none fired" })]));

  if (run.cluster_name) {
    $("#run-findings").append(el("p", { class: "muted",
      text: "cluster " + run.cluster_slug + ": " + run.cluster_name }));
  }

  const total = Math.max(run.wall_ms, 1);
  const tree = $("#run-tree");
  tree.replaceChildren(...run.spans.map((s) => {
    const left = (s.offset_ms / total) * 100;
    const width = Math.max((s.duration_ms / total) * 100, 0.4);
    const tokens = s.incl_tokens ? num(s.incl_tokens) + " tok" : "";
    const row = el("div", { class: "row" + (s.error ? " err" : "") }, [
      el("div", { class: "name", text: "  ".repeat(s.depth) + s.name, title: s.name }),
      el("div", { class: "n", text: s.duration_ms.toFixed(1) + " ms" }),
      el("div", { class: "n", text: tokens }),
      el("div", { class: "gantt" }, [el("i", { style: `left:${left}%;width:${width}%` })]),
    ]);
    row.addEventListener("click", () => showSpan(s));
    return row;
  }));
  $("#span-detail").replaceChildren();
}

function showSpan(span) {
  const lines = Object.entries(span.attributes)
    .map(([k, v]) => k + " = " + (typeof v === "object" ? JSON.stringify(v) : v));
  if (span.status_message) lines.unshift("status = " + span.status_message);
  lines.unshift("self " + span.self_ms.toFixed(1) + " ms, subtree cost " +
                usd(span.incl_cost_usd));
  $("#span-detail").replaceChildren(el("h3", { text: span.name }),
                                    el("pre", { text: lines.join("\n") }));
}

// --- clusters ----------------------------------------------------------------------------------

function renderClusters() {
  const clusters = state.clusters;
  const map = $("#cluster-map");
  map.replaceChildren();
  if (clusters.length) {
    const xs = clusters.map((c) => c.x), ys = clusters.map((c) => c.y);
    const spanX = (Math.max(...xs) - Math.min(...xs)) || 1;
    const spanY = (Math.max(...ys) - Math.min(...ys)) || 1;
    const biggest = Math.max(...clusters.map((c) => c.size)) || 1;
    for (const c of clusters) {
      const px = 8 + ((c.x - Math.min(...xs)) / spanX) * 84;
      const py = 90 - ((c.y - Math.min(...ys)) / spanY) * 80;
      const size = 10 + 44 * Math.sqrt(c.size / biggest);
      const incoherent = c.quality && c.quality.incoherent;
      const dot = el("div", {
        class: "dot", title: c.slug + ": " + c.name + " (n=" + c.size + ")",
        style: `left:${px}%;top:${py}%;width:${size}px;height:${size}px` +
               (incoherent ? ";background:var(--bad)" : ""),
      });
      dot.addEventListener("click", () => selectCluster(c.slug));
      map.append(dot, el("div", { class: "lbl", text: c.slug,
                                  style: `left:${px}%;top:${py}%` }));
    }
  }

  $("#clusters-body").replaceChildren(...clusters.map((c) => {
    const q = c.quality;
    const tr = el("tr", {}, [
      el("td", { class: "mono", text: c.slug }),
      el("td", {}, [el("span", { text: c.name }),
                    q && q.incoherent ? el("span", { class: "pill bad", text: "incoherent" }) : null]),
      el("td", { class: "n", text: num(c.size) }),
      el("td", { class: "n", text: c.cohesion.toFixed(2) }),
      el("td", { text: q ? Math.round(q.dominant_share * 100) + "% " + q.dominant_label : "-" }),
      el("td", { class: "muted mono",
                 text: c.top_features.slice(0, 3).map((f) => f[0].split("/").pop()).join(" ") }),
    ]);
    tr.addEventListener("click", () => selectCluster(c.slug));
    return tr;
  }));
}

function selectCluster(slug) {
  const c = state.clusters.find((x) => x.slug === slug);
  if (!c) return;
  for (const dot of document.querySelectorAll("#cluster-map .dot")) {
    dot.classList.toggle("sel", dot.title.startsWith(slug + ":"));
  }
  const detail = $("#cluster-detail");
  detail.replaceChildren(
    el("h3", { text: slug + "  " + c.name }),
    el("p", { text: c.summary }),
    el("pre", { text: c.top_features.map((f) => f[1].toFixed(3).padStart(7) + "  " + f[0])
      .join("\n") }),
    el("h3", { text: "Representative runs" }),
  );
  const list = el("div", { class: "controls" });
  for (const rid of c.representatives) {
    const b = el("button", { class: "act mono", text: rid });
    b.addEventListener("click", () => openRun(rid));
    list.append(b);
  }
  const all = el("button", { class: "act", text: "all " + c.size + " runs" });
  all.addEventListener("click", () => {
    $("#f-cluster").value = slug;
    show("runs");
    renderRuns();
  });
  detail.append(list, all);
}

// --- cost --------------------------------------------------------------------------------------

function barTable(headers, rows) {
  const peak = Math.max(...rows.map((r) => r.bar), 1);
  const table = el("table", {}, [
    el("thead", {}, [el("tr", {}, headers.map((h, i) =>
      el("th", { class: i ? "n" : "", text: h })))]),
    el("tbody", {}, rows.map((r) => el("tr", {}, [
      ...r.cells.map((c, i) => el("td", { class: i ? "n" : "mono", text: c })),
      el("td", {}, [el("div", { class: "bar" + (r.bad ? " bad" : ""),
                                style: `width:${(100 * r.bar) / peak}%` })]),
    ]))),
  ]);
  return el("div", { class: "scroll" }, [table]);
}

function renderCost() {
  const a = state.overview.attribution || {};
  const body = $("#cost-body");
  body.replaceChildren();
  const groups = [
    ["by_tool", "Tools by total time", "total_ms"],
    ["by_step", "Steps by inclusive cost", "cost_usd"],
    ["by_model", "Models by cost", "cost_usd"],
  ];
  for (const [key, title, sortKey] of groups) {
    const buckets = Object.entries(a[key] || {});
    if (!buckets.length) continue;
    buckets.sort((x, y) => (y[1][sortKey] || 0) - (x[1][sortKey] || 0));
    const rows = buckets.slice(0, 18).map(([name, b]) => ({
      bar: b[sortKey] || 0,
      bad: (b.error_rate || 0) > 0.2,
      cells: [name, num(b.calls), usd(b.cost_usd), num(b.tokens_in + b.tokens_out),
              num(b.mean_ms, 1), num(b.max_ms, 1), (100 * (b.error_rate || 0)).toFixed(1) + "%"],
    }));
    body.append(el("h3", { text: title }),
                barTable(["key", "calls", "cost", "tokens", "mean ms", "max ms", "errors", ""],
                         rows));
  }
}

// --- validation --------------------------------------------------------------------------------

function renderValidation() {
  const body = $("#validation-body");
  body.replaceChildren();
  const v = state.overview.validation;
  if (v) {
    body.append(statRow([
      ["purity", v.purity.toFixed(3)],
      ["inverse purity", v.inverse_purity.toFixed(3)],
      ["adjusted Rand", v.adjusted_rand.toFixed(3)],
      ["NMI", v.nmi.toFixed(3)],
      ["silhouette", v.silhouette.toFixed(3)],
      ["unclustered", (100 * v.noise_fraction).toFixed(1) + "%"],
      ["labelled", num(v.n_labelled)],
    ]));
    if (v.incoherent && v.incoherent.length) {
      body.append(el("p", { class: "bad",
        text: "Incoherent clusters: " + v.incoherent.join(", ") }));
    } else {
      body.append(el("p", { class: "good",
        text: "No cluster fell below 50% dominant label share." }));
    }
    const modes = Object.entries(v.mode_coverage || {}).sort((a, b) => b[1] - a[1]);
    body.append(el("h3", { text: "Planted mode recovery" }),
      barTable(["planted mode", "largest cluster holds", "clusters", ""],
        modes.map(([m, cov]) => ({
          bar: cov, bad: cov < 0.6,
          cells: [m, (100 * cov).toFixed(1) + "%", String((v.mode_split || {})[m] || 0)],
        }))));
  } else {
    body.append(el("p", { class: "muted",
      text: "No ground truth labels were supplied to obs analyse." }));
  }

  const d = state.overview.detectors;
  if (d) {
    body.append(el("h3", { text: "Detector precision" }),
      barTable(["pathology", "fired", "precision", "95% CI low", "recall", "F1", "false pos",
                "missed", ""],
        d.map((row) => ({
          bar: row.precision, bad: row.precision < 0.8,
          cells: [row.pathology, num(row.fired), row.precision.toFixed(3),
                  row.precision_ci[0].toFixed(2), row.recall.toFixed(3), row.f1.toFixed(3),
                  num(row.false_positives), num(row.missed)],
        }))));
  } else {
    const counts = Object.entries(state.overview.detector_counts || {}).sort((a, b) => b[1] - a[1]);
    body.append(el("h3", { text: "Detector findings (no labels, so no precision claimed)" }),
      barTable(["pathology", "runs", ""],
        counts.map(([p, n]) => ({ bar: n, bad: false, cells: [p, num(n)] }))));
  }
}

// --- regression --------------------------------------------------------------------------------

function renderRegression() {
  const body = $("#regression-body");
  body.replaceChildren();
  const all = state.overview.regressions || {};
  const keys = Object.keys(all);
  if (!keys.length) {
    body.append(el("p", { class: "muted",
      text: "Run obs regress --before X --after Y to populate this." }));
    return;
  }
  for (const key of keys) {
    const r = all[key];
    body.append(el("h3", { text: key + "   n=" + r.n_a + " vs " + r.n_b +
                                  "   BH alpha " + r.alpha }));
    const grew = r.clusters.filter((d) => d.significant && d.diff > 0);
    body.append(el("p", { class: grew.length ? "bad" : "good",
      text: grew.length
        ? "Regressed: " + grew.map((d) => d.key + " (" + d.name + ") x" +
            d.risk_ratio.toFixed(2)).join(", ")
        : "No cluster grew significantly." }));
    for (const [title, rows] of [["clusters", r.clusters], ["pathologies", r.pathologies]]) {
      if (!rows || !rows.length) continue;
      body.append(el("h3", { text: title }),
        barTable(["key", "name", r.label_a, r.label_b, "RR", "RR low", "RR high", "p", "q",
                  "verdict", ""],
          rows.map((d) => ({
            bar: Math.abs(d.diff), bad: d.significant && d.diff > 0,
            cells: [d.key, d.name, (100 * d.rate_a).toFixed(2) + "%",
                    (100 * d.rate_b).toFixed(2) + "%", d.risk_ratio.toFixed(2),
                    d.rr_ci[0].toFixed(2), d.rr_ci[1].toFixed(2), d.p_value.toFixed(4),
                    d.p_adjusted.toFixed(4), d.direction],
          }))));
    }
  }
}

// --- boot --------------------------------------------------------------------------------------

(async function boot() {
  state.overview = await get("/api/overview");
  state.clusters = await get("/api/clusters");
  const s = state.overview.summary || {};
  $("#head-stat").textContent =
    `${num(s.n_runs)} runs, ${num(s.n_spans)} spans, ${num(s.k)} clusters, ` +
    `${usd(s.total_cost_usd)}`;
  fillSelect($("#f-cluster"), "any cluster", state.clusters.map((c) => c.slug));
  fillSelect($("#f-cohort"), "any cohort", state.overview.cohorts || []);
  fillSelect($("#f-pathology"), "any detector",
             Object.keys(state.overview.detector_counts || {}).sort());
  await renderRuns();
})();
