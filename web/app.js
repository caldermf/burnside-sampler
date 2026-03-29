const elements = {
  q: document.querySelector("#prime-q"),
  n: document.querySelector("#dimension-n"),
  steps: document.querySelector("#step-count"),
  seed: document.querySelector("#seed"),
  start: document.querySelector("#start-state"),
  initializeButton: document.querySelector("#initialize-button"),
  runButton: document.querySelector("#run-button"),
  stepButton: document.querySelector("#step-button"),
  resetButton: document.querySelector("#reset-button"),
  status: document.querySelector("#status-text"),
  current: document.querySelector("#metric-current"),
  visited: document.querySelector("#metric-visited"),
  stepsMetric: document.querySelector("#metric-steps"),
  mode: document.querySelector("#metric-mode"),
  history: document.querySelector("#history-strip"),
  topStates: document.querySelector("#top-states"),
  matrix: document.querySelector("#matrix-view"),
  trace: document.querySelector("#trace-view"),
  chart: document.querySelector("#distribution-chart"),
};

const worker = new Worker(new URL("./worker.mjs", import.meta.url), { type: "module" });

let workerReady = false;
let busy = false;
let latestSnapshot = null;

function permutationLabel(permutation) {
  return `[${permutation.join(", ")}]`;
}

function isPrime(n) {
  if (n < 2) {
    return false;
  }
  if (n % 2 === 0) {
    return n === 2;
  }
  for (let factor = 3; factor * factor <= n; factor += 2) {
    if (n % factor === 0) {
      return false;
    }
  }
  return true;
}

function parsePermutation(text, n) {
  const tokens = text
    .trim()
    .split(/[\s,]+/)
    .filter(Boolean)
    .map((value) => Number(value));

  if (tokens.length === 0) {
    return Array.from({ length: n }, (_, index) => index + 1);
  }
  if (tokens.length !== n) {
    throw new Error(`Start permutation must contain exactly ${n} entries.`);
  }
  const sorted = [...tokens].sort((left, right) => left - right);
  const target = Array.from({ length: n }, (_, index) => index + 1);
  if (!sorted.every((value, index) => value === target[index])) {
    throw new Error(`Start permutation must be a rearrangement of 1 through ${n}.`);
  }
  return tokens;
}

function setBusy(nextBusy) {
  busy = nextBusy;
  for (const button of [
    elements.initializeButton,
    elements.runButton,
    elements.stepButton,
    elements.resetButton,
  ]) {
    button.disabled = nextBusy || !workerReady;
  }
}

function setStatus(message) {
  elements.status.textContent = message;
}

function formatMatrix(rows) {
  if (!rows || rows.length === 0) {
    return "No quotient at this stage.";
  }
  const width = Math.max(...rows.flat().map((value) => String(value).length));
  return rows
    .map((row) => row.map((value) => String(value).padStart(width, " ")).join("  "))
    .join("\n");
}

function drawChart(snapshot) {
  const canvas = elements.chart;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width * dpr));
  const height = Math.max(260, Math.floor(rect.height * dpr));
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const cssWidth = width / dpr;
  const cssHeight = height / dpr;

  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = "#faf7f0";
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const entries = (snapshot?.topStates ?? []).slice(0, 12);
  const margin = { top: 24, right: 18, bottom: 72, left: 56 };
  const chartWidth = cssWidth - margin.left - margin.right;
  const chartHeight = cssHeight - margin.top - margin.bottom;

  ctx.strokeStyle = "rgba(31, 38, 34, 0.08)";
  ctx.lineWidth = 1;
  const maxCount = Math.max(1, ...entries.map((entry) => entry.count));
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = margin.top + (chartHeight * tick) / 4;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + chartWidth, y);
    ctx.stroke();
    const value = Math.round(maxCount * (1 - tick / 4));
    ctx.fillStyle = "#5b645d";
    ctx.font = '11px "IBM Plex Mono"';
    ctx.fillText(String(value), 10, y + 4);
  }

  if (entries.length === 0) {
    ctx.fillStyle = "#5b645d";
    ctx.font = '15px "IBM Plex Mono"';
    ctx.fillText("Run the chain to populate the visit distribution.", margin.left, margin.top + 32);
    return;
  }

  const barGap = 10;
  const barWidth = (chartWidth - barGap * (entries.length - 1)) / entries.length;

  entries.forEach((entry, index) => {
    const x = margin.left + index * (barWidth + barGap);
    const barHeight = (entry.count / maxCount) * chartHeight;
    const y = margin.top + chartHeight - barHeight;

    ctx.fillStyle = "#d7e0d9";
    ctx.fillRect(x, y, barWidth, barHeight);

    ctx.fillStyle = "#234b39";
    ctx.fillRect(x, y + Math.max(0, barHeight - 6), barWidth, Math.min(6, barHeight));

    ctx.save();
    ctx.translate(x + barWidth / 2, margin.top + chartHeight + 18);
    ctx.rotate(-Math.PI / 5.5);
    ctx.fillStyle = "#1f2622";
    ctx.font = '11px "IBM Plex Mono"';
    ctx.textAlign = "right";
    ctx.fillText(permutationLabel(entry.permutation), 0, 0);
    ctx.restore();
  });
}

function renderTopStates(snapshot) {
  const rows = snapshot.topStates ?? [];
  if (rows.length === 0) {
    elements.topStates.innerHTML = '<div class="empty-state">No visit counts yet.</div>';
    return;
  }

  const body = rows
    .map(
      (entry, index) => `
        <tr>
          <td class="mono">${index + 1}</td>
          <td class="mono">${permutationLabel(entry.permutation)}</td>
          <td class="mono">${entry.count}</td>
        </tr>`,
    )
    .join("");

  elements.topStates.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Permutation</th>
          <th>Visits</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderHistory(snapshot) {
  const history = snapshot.history ?? [];
  if (history.length === 0) {
    elements.history.innerHTML = '<div class="empty-state">No trajectory yet.</div>';
    return;
  }

  elements.history.innerHTML = history
    .map((state) => `<span class="history-pill">${permutationLabel(state)}</span>`)
    .join("");
}

function renderTrace(snapshot) {
  const latest = snapshot.latest;
  if (!latest) {
    elements.matrix.textContent = "No step sampled yet.";
    elements.trace.innerHTML = '<div class="empty-state">Initialize the session and take a step to inspect the recursive quotient trace.</div>';
    return;
  }

  elements.matrix.textContent = formatMatrix(latest.stabilizer);
  const traceMarkup = latest.trace
    .map((entry) => {
      const weights = entry.weights.length
        ? entry.weights
            .map((item) => `${permutationLabel(item.partition)} : ${item.weight}`)
            .join("  |  ")
        : "Base case";

      return `
        <article class="trace-step">
          <div class="trace-step-head">
            <h3>dim ${entry.dimension}</h3>
            <span class="card-note mono">${permutationLabel(entry.partition)}</span>
          </div>
          <div class="trace-meta">
            Chosen smaller partition: ${entry.smallerPartition.length ? permutationLabel(entry.smallerPartition) : "[]"}<br>
            First vector: ${permutationLabel(entry.firstVector)}
          </div>
          <div class="trace-weights">Weights: ${weights}</div>
        </article>
      `;
    })
    .join("");

  elements.trace.innerHTML = traceMarkup;
}

function renderSnapshot(snapshot) {
  latestSnapshot = snapshot;
  elements.current.textContent = permutationLabel(snapshot.current);
  elements.visited.textContent = `${snapshot.visited} / ${snapshot.totalStates}`;
  elements.stepsMetric.textContent = String(snapshot.steps);
  elements.mode.textContent = `${permutationLabel(snapshot.mode.permutation)} (${snapshot.mode.count})`;
  renderHistory(snapshot);
  renderTopStates(snapshot);
  renderTrace(snapshot);
  drawChart(snapshot);
}

function readControls() {
  const n = Number(elements.n.value);
  const q = Number(elements.q.value);
  const steps = Number(elements.steps.value);
  const seed = elements.seed.value === "" ? null : Number(elements.seed.value);
  const start = parsePermutation(elements.start.value, n);

  if (!Number.isInteger(q) || q < 2) {
    throw new Error("q must be an integer prime.");
  }
  if (!isPrime(q)) {
    throw new Error("q must be prime in the current browser implementation.");
  }
  if (!Number.isInteger(n) || n < 2 || n > 7) {
    throw new Error("n must be between 2 and 7 for the browser view.");
  }
  if (!Number.isInteger(steps) || steps < 1) {
    throw new Error("Steps must be a positive integer.");
  }

  return { q, n, steps, seed, start };
}

function postWorkerMessage(message, statusText) {
  setBusy(true);
  setStatus(statusText);
  worker.postMessage(message);
}

function initializeSession() {
  const controls = readControls();
  postWorkerMessage(
    { type: "init", ...controls },
    `Initializing q=${controls.q}, n=${controls.n}…`,
  );
}

function resetIdentity() {
  const n = Number(elements.n.value);
  const start = Array.from({ length: n }, (_, index) => index + 1);
  elements.start.value = start.join(" ");
  if (!workerReady) {
    return;
  }
  postWorkerMessage({ type: "reset", start }, "Resetting to the identity permutation…");
}

worker.addEventListener("message", (event) => {
  const { type, payload, message } = event.data ?? {};

  if (type === "ready") {
    workerReady = true;
    setBusy(false);
    setStatus(message);
    initializeSession();
    return;
  }

  if (type === "snapshot") {
    renderSnapshot(payload);
    setBusy(false);
    setStatus(message);
    return;
  }

  if (type === "error") {
    setBusy(false);
    setStatus(`Error: ${message}`);
  }
});

elements.initializeButton.addEventListener("click", () => {
  try {
    initializeSession();
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error));
  }
});

elements.runButton.addEventListener("click", () => {
  try {
    const controls = readControls();
    postWorkerMessage({ type: "run", steps: controls.steps }, `Running ${controls.steps} steps…`);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error));
  }
});

elements.stepButton.addEventListener("click", () => {
  postWorkerMessage({ type: "step" }, "Advancing one step…");
});

elements.resetButton.addEventListener("click", () => {
  try {
    resetIdentity();
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error));
  }
});

elements.n.addEventListener("change", () => {
  const n = Number(elements.n.value);
  if (Number.isInteger(n) && n >= 2) {
    elements.start.value = Array.from({ length: n }, (_, index) => index + 1).join(" ");
  }
});

window.addEventListener("resize", () => {
  if (latestSnapshot) {
    drawChart(latestSnapshot);
  }
});

setBusy(true);
