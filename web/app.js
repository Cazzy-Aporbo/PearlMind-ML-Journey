"use strict";
const $ = (id) => document.getElementById(id);
const motion = $("motion");
motion.addEventListener("click", () => {
  const off = document.body.classList.toggle("motion-off");
  motion.setAttribute("aria-pressed", String(off));
  motion.textContent = off ? "Resume motion" : "Pause motion";
});
let w = 0,
  iteration = 0,
  trace = [];
const loss = (x) => (14 / 3) * (x - 2) ** 2;
const point = (x) => [
  40 + ((x + 1) * 435) / 6,
  265 - (Math.min(loss(x), 45) * 240) / 45,
];
$("loss-curve").setAttribute(
  "d",
  Array.from({ length: 121 }, (_, i) => {
    const [x, y] = point(-1 + i * 0.05);
    return `${i ? "L" : "M"}${x},${y}`;
  }).join(" "),
);
function showGradient() {
  const [x, y] = point(w);
  $("weight-dot").setAttribute("cx", Math.max(40, Math.min(475, x)));
  $("weight-dot").setAttribute("cy", y);
  $("weight").textContent = w.toFixed(3);
  $("loss").textContent = loss(w).toFixed(3);
  $("iteration").textContent = iteration;
  $("trace").setAttribute(
    "d",
    trace
      .map((v, i) => {
        const [a, b] = point(v);
        return `${i ? "L" : "M"}${Math.max(40, Math.min(475, a))},${b}`;
      })
      .join(" "),
  );
  $("gradient-note").textContent =
    Math.abs(w) > 6
      ? "The update is leaving the plot. Reduce the learning rate and restart."
      : "Synthetic points: (1,2), (2,4), (3,6). Model: ŷ = wx.";
}
$("step").addEventListener("click", () => {
  if (Math.abs(w) > 1e4) return;
  trace.push(w);
  w -= ((Number($("rate").value) * 28) / 3) * (w - 2);
  iteration++;
  trace.push(w);
  showGradient();
});
$("reset").addEventListener("click", () => {
  w = 0;
  iteration = 0;
  trace = [];
  showGradient();
});
$("rate").addEventListener(
  "input",
  () => ($("rate-value").textContent = Number($("rate").value).toFixed(2)),
);
showGradient();
const observations = [
  { p: 0.08, y: 0 },
  { p: 0.15, y: 0 },
  { p: 0.24, y: 1 },
  { p: 0.32, y: 0 },
  { p: 0.42, y: 1 },
  { p: 0.49, y: 0 },
  { p: 0.55, y: 1 },
  { p: 0.64, y: 0 },
  { p: 0.72, y: 1 },
  { p: 0.81, y: 1 },
  { p: 0.9, y: 0 },
  { p: 0.96, y: 1 },
];
function threshold() {
  const t = Number($("cutoff").value),
    cost = Number($("cost").value);
  $("cutoff-value").textContent = t.toFixed(2);
  $("cost-value").textContent = cost;
  let tp = 0,
    tn = 0,
    fp = 0,
    fn = 0;
  $("observations").replaceChildren();
  observations.forEach((o, i) => {
    const pred = Number(o.p >= t);
    if (pred && o.y) tp++;
    else if (pred) fp++;
    else if (o.y) fn++;
    else tn++;
    const item = document.createElement("span");
    item.className = "observation" + (pred !== o.y ? " error" : "");
    item.title = `Observation ${i + 1}: true class ${o.y}, probability ${o.p}, predicted ${pred}`;
    item.textContent = o.y ? "●" : "○";
    const label = document.createElement("small");
    label.textContent = o.p.toFixed(2);
    item.append(label);
    $("observations").append(item);
  });
  $("matrix").replaceChildren();
  [
    ["True positive", tp],
    ["False positive", fp],
    ["False negative", fn],
    ["True negative", tn],
  ].forEach(([name, n]) => {
    const el = document.createElement("span");
    el.textContent = name;
    const strong = document.createElement("strong");
    strong.textContent = n;
    el.append(strong);
    $("matrix").append(el);
  });
  $("threshold-note").textContent =
    `Precision: ${tp + fp ? (tp / (tp + fp)).toFixed(2) : "undefined"} · Recall: ${(tp / (tp + fn)).toFixed(2)} · Illustrative cost: ${fp + cost * fn}`;
}
$("cutoff").addEventListener("input", threshold);
$("cost").addEventListener("input", threshold);
threshold();
function workers() {
  const n = Number($("workers").value);
  $("workers-value").textContent = n;
  $("worker-count").textContent = `${n} examples`;
  $("aggregate").textContent = ((-8 - 10 * n) / (2 + n)).toFixed(3);
  $("formula").textContent = `(2 × −4 + ${n} × −10) / ${2 + n}`;
}
$("workers").addEventListener("input", workers);
workers();
async function loadEvidence() {
  try {
    const r = await fetch("data/evidence.json");
    if (!r.ok) throw Error("missing evidence");
    const d = await r.json();
    const node = $("evidence-content");
    node.replaceChildren();
    [
      [
        "Tree · held-out accuracy",
        `${(d.tabular.overall_accuracy * 100).toFixed(1)}%`,
      ],
      [
        "Neural · held-out accuracy",
        `${(d.torch.test_accuracy * 100).toFixed(1)}%`,
      ],
      ["Neural · trainable parameters", d.torch.parameters],
      [
        "Weighted gradient equality",
        d.systems.gradient_aggregation.matches_full_batch
          ? "verified"
          : "failed",
      ],
    ].forEach(([title, value]) => {
      const el = document.createElement("div"),
        label = document.createElement("span"),
        b = document.createElement("b");
      label.textContent = title;
      b.textContent = value;
      el.append(label, b);
      node.append(el);
    });
    const p = document.createElement("p");
    p.className = "fine";
    p.textContent = `Measured ${d.created_at} · commit ${d.commit.slice(0, 8)} · ${d.python}. Single synthetic split; see JSON for seeds, shapes, losses and limitations.`;
    node.append(p);
  } catch (e) {
    $("evidence-content").textContent =
      "No measured run is bundled with this preview. Run scripts/run_evidence.py before building; no scores are substituted.";
  }
}
loadEvidence();
let catalog = [],
  fileLimit = 15;
function renderFiles() {
  const query = $("search").value.toLowerCase(),
    area = $("area").value;
  const matches = catalog.filter(
    (f) =>
      (!area || f.area === area) &&
      `${f.path} ${f.description} ${f.imports.join(" ")}`
        .toLowerCase()
        .includes(query),
  );
  $("count").textContent = `${matches.length} files`;
  $("file-list").replaceChildren();
  $("more-files").hidden = matches.length <= fileLimit;
  matches.slice(0, fileLimit).forEach((f) => {
    const row = document.createElement("div");
    row.className = "file-row";
    const a = document.createElement("a");
    a.href = f.page;
    a.textContent = f.path + " ↗";
    const p = document.createElement("p");
    p.textContent = f.description;
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = f.status;
    row.append(a, p, tag);
    $("file-list").append(row);
  });
}
fetch("data/catalog.json")
  .then((r) => {
    if (!r.ok) throw Error("catalog");
    return r.json();
  })
  .then((files) => {
    catalog = files;
    [...new Set(files.map((f) => f.area))].sort().forEach((area) => {
      const option = document.createElement("option");
      option.value = area;
      option.textContent = area;
      $("area").append(option);
    });
    renderFiles();
  })
  .catch(() => {
    $("file-list").textContent = "Build the site to generate the source atlas.";
  });
$("search").addEventListener("input", () => {
  fileLimit = 15;
  renderFiles();
});
$("area").addEventListener("change", () => {
  fileLimit = 15;
  renderFiles();
});
$("more-files").addEventListener("click", () => {
  fileLimit += 15;
  renderFiles();
});
