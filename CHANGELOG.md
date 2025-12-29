# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Chores

- add uv-managed dependencies (43a15cb)

- configure pre-commit and format code (53e0837)

- add dvc setup and dvc pull helper (4c69049)

- track cassava dataset with dvc (7528556)

- add docker-compose for local mlflow (c775da6)

- add git-cliff changelog config (9b70d6d)

- update changelog (7ebfc8d)

- configure yandex s3 remote bucket and prefix (bf7562d)

- ignore dvc local config (beae76c)

- add github actions ci (d27fd54)

- add public http remote for no-cred pulls (2532d22)

- remove obsolete s3 env example (a4230ed)

- cleanup dependencies and update lock (089e559)

- add fire CLI dependency and scripts (edac026)

- remove warnings.filterwarnings ignores (ae81b42)

- enable ruff pydocstyle (google) (4251084)

- add mypy (d9144bb)

### Docs

- expand README for task2 workflow (4736d10)

- rewrite onboarding (RU+EN) (c1ca43f)

- align readme and proposal with project description (7e4c452)

- clarify uv run --no-sync for CUDA torch (0591feb)

- rewrite onboarding instructions (24832b1)

### Features

- add python package skeleton and CLI entrypoint (07d66e4)

- add hydra config hierarchy (52b9192)

- add lightning training pipeline (30ceb0a)

- add stratified kfold mode and f1-macro logging (d69da18)

- add minimal fastapi inference app (9206132)

- add dataset sample limits for quick runs (c7ebd26)

- add infer command (edbae85)

- harden cli and add optional artifact uploads (bd0c822)

- add simple multirun for train (3359e41)

- add download-data for public dataset (37c4df1)

- add transfer learning freeze/unfreeze (4c4268c)

- add imbalance handling, strong augmentations, and LR schedulers (26a6f9b)

- add max_time limit, S3 metrics upload, and helper tests (2c964c9)

- add download-model and improve checkpoint handling (39c5e35)

### Fixes

- enforce utf-8 stdio for mlflow on windows (5f2f077)

- improve fire wrapper UX (8c36dd3)

- support rar archives in download-data (00c344c)

- fail fast when gpu requested without CUDA (9b1fd28)

- make fire positional args map to hydra overrides (3c517f2)

- normalize max_time to DD:HH:MM:SS for Lightning Timer (2882636)

### refactor

- move remaining magic constants to hydra (e12a9ec)

- use hydra compose API from fire wrapper (8056d64)

### revert

- drop pydantic schemas (cdac3d1)

### style

- replace single-letter variables with semantic names (ad7c68f)

### test

- restructure tests and raise coverage (eaac270)
