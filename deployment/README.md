The canonical locks and installer modules live in
`benchmark/src/piper_bench/resources/deployment` and `piper_bench.deployment`.
The current deployment uses the command-line installer for separate native
simulation and writer environments from the packaged locks and asset manifest.
Optional EWM/OpenWAM and FastWAM profiles are outside this deployment scope and
have not passed fresh-environment acceptance.
See [deployment instructions](../docs/deployment.md) for prerequisites,
license acceptance, validation scopes, and known metadata exceptions.

Run `./ewm setup --profile <profile> --prefix <new-managed-prefix>` from the
repository, or use `deployment/install.sh` with the same arguments. Python
packages, Git revisions, compatibility code and runtime assets are pinned by the
release lock; host drivers and FFmpeg/system libraries remain explicit system
prerequisites. Containers are outside the requested deployment scope.

The source release does not need the root `migration/`, `Event-DataCollection/`,
`assets/` or `archive/` directories. Small runtime assets and dependency/source
locks are shipped inside `benchmark`. `runtime_env.sh` selects existing local
simulation/writer/EWM interpreters for experiment launchers; `activate_training.sh`
selects an existing FastWAM/GR00T environment. Neither helper installs packages.
New installations use `ewm setup` and its generated `PIPER_CONFIG`.
