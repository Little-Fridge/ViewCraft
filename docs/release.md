# Release scope

The repository includes the ViewCraft runtime and joint reader/interface evolution framework, custom-task and scoring support, prediction, resume, benchmark adapters, and a minimal synthetic counting example.

`skill-library/` contains ten hand-authored reusable task skills. They are starting configurations, not automatically evolved checkpoints or a reproduction of paper results.

`website/` contains the paper project page, original paper figures and the preprint PDF. Paper results shown there are separate from the starter skill library. See [website publishing](website.md).

## Reproduction boundary

This release does not contain the original benchmark datasets or archived paper experiment runs. Demo defaults and historical benchmark configurations do not reproduce every paper protocol or ablation. Follow each dataset's access and license terms.

API credentials, private provider routes, downloaded benchmark data, caches, model weights and run directories are excluded. Users can record their own executions locally.

Offline tests check the execution path, skill configurations, input handling and recording fidelity. They do not establish real-model accuracy. Run them with `python -m pytest -q` after installing `requirements-dev.txt`.
