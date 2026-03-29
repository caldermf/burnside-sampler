# burnside-sampler

This repo now has two trusted layers:

- [`sampling/sampling_burnside.ipynb`](/Users/caldermf/projects/burnside-sampler/sampling/sampling_burnside.ipynb), the original Sage notebook and source-of-truth oracle.
- [`burnside_sampler/pure_python.py`](/Users/caldermf/projects/burnside-sampler/burnside_sampler/pure_python.py), a pure-Python port restricted to prime fields.

The Sage-backed oracle wrapper lives in [`burnside_sampler/oracle_sage.py`](/Users/caldermf/projects/burnside-sampler/burnside_sampler/oracle_sage.py). It exists so the pure-Python code can be tested directly against the notebook logic.

Run the verification suite with:

```sh
./run_tests.sh
```

That runs:

- the pure-Python correctness tests under `python`
- the oracle-alignment tests under `sage -python`

The static browser app lives in [`web/index.html`](/Users/caldermf/projects/burnside-sampler/web/index.html).
Serve the repo root and open `/web/index.html`:

```sh
./serve_web.sh
```

Then visit:

```txt
http://localhost:8000/web/index.html
```

The frontend runs entirely in the browser through Pyodide in a web worker and uses the verified prime-field sampler from [`burnside_sampler/pure_python.py`](/Users/caldermf/projects/burnside-sampler/burnside_sampler/pure_python.py).
