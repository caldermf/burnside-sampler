# Burnside Sampler

Static browser visualizer and verified prime-field implementation of the Burnside sampling process on the symmetric group.

The project has two core layers:

- `web/`: a zero-backend browser app that runs the sampler locally through Pyodide
- `burnside_sampler/`: the tested pure-Python implementation, together with a Sage oracle used for correctness checks

The mathematical source of truth is still the original Sage notebook:

- `sampling/sampling_burnside.ipynb`

## What This Repo Contains

- `burnside_sampler/pure_python.py`
  Prime-field Burnside sampler used in the browser app.
- `burnside_sampler/oracle_sage.py`
  Sage-backed reference extracted from the notebook for alignment tests.
- `tests/`
  Correctness and oracle-alignment tests.
- `web/`
  Public-facing static app.
- `run_tests.sh`
  Runs the Python tests and the Sage-backed verification suite.
- `serve_web.sh`
  Serves the repository locally for browser use.

## Browser App

The web app is fully static. There is no server-side simulation layer and no backend state.

To run it locally:

```sh
./serve_web.sh
```

Then open:

```txt
http://localhost:8000/web/index.html
```

The browser UI:

- simulates the Burnside chain locally
- restricts `q` to primes
- colors histogram bars by right Steinberg cell via the RSK insertion tableau `P`
- keeps the mathematical kernel in Python rather than duplicating it in JavaScript

## Verification

Run the full verification suite with:

```sh
./run_tests.sh
```

This checks:

- pure-Python finite-field linear algebra and Green-polynomial logic
- quotient and Springer-fiber sampling behavior
- alignment against Sage for the trusted oracle
- RSK / right-cell metadata used by the visualization

## Project Goal

This repository is organized around one constraint: the visualizer should be easy to host and impressive to use without compromising mathematical correctness.

The intended workflow is:

1. treat `sampling/sampling_burnside.ipynb` as the oracle
2. verify changes against Sage
3. expose only the verified pure-Python kernel in the browser

## Notes

- The current implementation is restricted to prime fields `F_p`.
- The frontend is designed for static hosting.
- Sage is needed only for oracle verification, not for running the public web app.
