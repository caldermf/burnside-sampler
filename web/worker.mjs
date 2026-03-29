import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.mjs";

const PYODIDE_VERSION = "0.29.3";
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let pyodide = null;
let ready = false;

async function installPythonModule() {
  const moduleUrl = new URL("../burnside_sampler/pure_python.py", import.meta.url);
  const source = await fetch(moduleUrl).then((response) => {
    if (!response.ok) {
      throw new Error(`Could not load ${moduleUrl.href}`);
    }
    return response.text();
  });

  pyodide.FS.mkdirTree("/burnside_sampler");
  pyodide.FS.writeFile(
    "/burnside_sampler/__init__.py",
    "from .pure_python import PrimeFieldBurnsideSampler\n",
  );
  pyodide.FS.writeFile("/burnside_sampler/pure_python.py", source);

  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/")
`);
}

async function bootstrapBridge() {
  await pyodide.runPythonAsync(`
import itertools
import math
import random

from burnside_sampler.pure_python import PrimeFieldBurnsideSampler


class BrowserSession:
    def __init__(self, q, n, start=None, seed=None):
        self.q = int(q)
        self.n = int(n)
        self.seed = None if seed is None else int(seed)
        self.sampler = PrimeFieldBurnsideSampler(self.q)
        self.rng = random.Random(self.seed)
        self.all_states = list(itertools.permutations(range(1, self.n + 1)))
        self._identity = tuple(range(1, self.n + 1))
        self.current = tuple(start) if start is not None else self._identity
        self.history = [self.current]
        self.counts = {state: 0 for state in self.all_states}
        self.counts[self.current] = 1
        self.last_details = None

    def _mode(self):
        state, count = max(self.counts.items(), key=lambda item: (item[1], item[0]))
        return {"permutation": list(state), "count": count}

    def _top_states(self, limit=24):
        ranked = sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"permutation": list(state), "count": count}
            for state, count in ranked[:limit]
            if count > 0
        ]

    def _recent_history(self, limit=16):
        return [list(state) for state in self.history[-limit:]]

    def _serialize_trace(self, trace):
        if trace is None:
            return []
        serialized = []
        for entry in trace:
            weights = [
                {"partition": list(partition), "weight": int(weight)}
                for partition, weight in sorted(
                    entry["smaller_weights"].items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            serialized.append(
                {
                    "dimension": entry["dimension"],
                    "partition": list(entry["partition"]),
                    "smallerPartition": list(entry["smaller_partition"]),
                    "weights": weights,
                    "firstVector": list(entry["first_vector"]),
                    "matrix": [list(row) for row in entry["matrix"]],
                    "quotient": [list(row) for row in entry["quotient"]],
                }
            )
        return serialized

    def _latest(self):
        if self.last_details is None:
            return None
        return {
            "current": list(self.last_details["current"]),
            "next": list(self.last_details["next"]),
            "stabilizer": [list(row) for row in self.last_details["stabilizer"]],
            "flag": [list(vector) for vector in self.last_details["flag"]],
            "trace": self._serialize_trace(self.last_details["trace"]),
        }

    def snapshot(self):
        return {
            "q": self.q,
            "n": self.n,
            "seed": self.seed,
            "current": list(self.current),
            "steps": len(self.history) - 1,
            "visited": sum(1 for count in self.counts.values() if count > 0),
            "totalStates": math.factorial(self.n),
            "mode": self._mode(),
            "topStates": self._top_states(),
            "history": self._recent_history(),
            "latest": self._latest(),
        }

    def step(self):
        self.last_details = self.sampler.next_step_with_details(self.current, self.rng)
        self.current = tuple(self.last_details["next"])
        self.history.append(self.current)
        self.counts[self.current] += 1
        return self.snapshot()

    def run(self, steps):
        total_steps = int(steps)
        if total_steps < 1:
            raise ValueError("Steps must be a positive integer.")
        for _ in range(total_steps):
            self.step()
        return self.snapshot()

    def reset(self, start=None):
        self.current = tuple(start) if start is not None else self._identity
        self.history = [self.current]
        self.counts = {state: 0 for state in self.all_states}
        self.counts[self.current] = 1
        self.last_details = None
        return self.snapshot()


SESSION = None


def create_session(q, n, start=None, seed=None):
    global SESSION
    SESSION = BrowserSession(q, n, start, seed)
    return SESSION.snapshot()


def step_session():
    if SESSION is None:
        raise RuntimeError("The browser session is not initialized yet.")
    return SESSION.step()


def run_session(steps):
    if SESSION is None:
        raise RuntimeError("The browser session is not initialized yet.")
    return SESSION.run(steps)


def reset_session(start=None):
    if SESSION is None:
        raise RuntimeError("The browser session is not initialized yet.")
    return SESSION.reset(start)
`);
}

async function ensureReady() {
  if (ready) {
    return;
  }
  pyodide = await loadPyodide({ indexURL: INDEX_URL });
  await installPythonModule();
  await bootstrapBridge();
  ready = true;
}

async function callPython(name, ...args) {
  const fn = pyodide.globals.get(name);
  try {
    const result = fn(...args);
    const data = result.toJs({ dict_converter: Object.fromEntries });
    result.destroy?.();
    return data;
  } finally {
    fn.destroy?.();
  }
}

async function handleMessage(event) {
  try {
    await ensureReady();

    if (event.data?.type === "init") {
      const payload = await callPython(
        "create_session",
        event.data.q,
        event.data.n,
        event.data.start ?? null,
        event.data.seed ?? null,
      );
      self.postMessage({ type: "snapshot", payload, message: "Session initialized." });
      return;
    }

    if (event.data?.type === "step") {
      const payload = await callPython("step_session");
      self.postMessage({ type: "snapshot", payload, message: "Advanced one Burnside step." });
      return;
    }

    if (event.data?.type === "run") {
      const payload = await callPython("run_session", event.data.steps);
      self.postMessage({ type: "snapshot", payload, message: `Ran ${event.data.steps} steps.` });
      return;
    }

    if (event.data?.type === "reset") {
      const payload = await callPython("reset_session", event.data.start ?? null);
      self.postMessage({ type: "snapshot", payload, message: "Session reset." });
      return;
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
}

self.addEventListener("message", handleMessage);

ensureReady()
  .then(() => {
    self.postMessage({
      type: "ready",
      message: `Pyodide ${PYODIDE_VERSION} loaded.`,
    });
  })
  .catch((error) => {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  });
