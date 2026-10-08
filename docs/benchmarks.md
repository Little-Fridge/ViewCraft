# Existing benchmark adapters

For your own task, start with `scripts/evolve.py`; benchmark adapters are useful
when you already have a dataset in an adapter's expected layout.

```bash
python scripts/run_domain.py --help
```

This lists supported domain names. Adapters live in `viewcraft/adapters/`;
`registry.py` records data subdirectories, scorer identities, and answer styles.
Dataset files are not bundled. Inspect the relevant loader for its input layout.
PDF domains need a rasterizer from `requirements-docs.txt`; video domains may
require ffmpeg. A list of adapters is not a claim that every external dataset has
been tested in this release environment.

After preparing the data and setting your models:

```bash
export VIEWCRAFT_DATA_ROOT=/path/to/benchmark/data
export VIEWCRAFT_CONFIG=configs/experiment.yaml
python scripts/run_domain.py --domain funsd --run-id funsd-demo --stage evolve
python scripts/run_domain.py --domain funsd --run-id funsd-demo --stage test
```

Pass `--initial-skill /path/to/skill` to replace built-in initialization.
`VIEWCRAFT_RUNS_ROOT` applies to both entry points and monitoring tools.
The benchmark runner retains legacy experiment options; its manifest is less
strict than the custom-task runner. Use a fresh run ID when changing inputs.

This release aligns FUNSD field-value scoring with the submission's normalized
edit similarity and gives it a new scorer ID. It does not bundle archived split
manifests, per-item paper predictions, or all paper comparison runners. Use the
task examples to learn the framework rather than treating demo defaults as the
paper's frozen experiment protocol.
