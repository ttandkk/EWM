# CVPR2027 independent-account migration snapshot

Read [AGENTS.md](AGENTS.md) first. This repository preserves the current first-party working trees of OpenWAM and Event-DataCollection, including EWM memory and Piper conveyor code that was not committed in the source repositories.

## Deployment

1. Choose a writable NEW_PROJECT_ROOT under the destination account and clone this repository there.
2. Review migration/sources.json and run `python3 migration/restore_sources.py --root "$NEW_PROJECT_ROOT" --dry-run`, then restore the pinned sources.
3. Fetch necessary LFS assets and independently install OpenWAM (Python3.10), Isaac Sim6.0.1 (Python3.12), and LeRobot writer (Python3.12). See AGENTS.md and migration/dependencies/.
4. Transfer datasets, pretrained weights and selected policy checkpoints separately. They are deliberately excluded from Git.
5. Replace source-account paths in launchers/configuration and validate imports, real data, simulation, collection, training and closed-loop evaluation.

Dependency snapshots document the existing installation; they are not tested installation lockfiles. The destination environment has not been installed or tested.

## Source provenance

- OpenWAM source HEAD: f6d9f1059eb63a8a76dc60e07da2dad21615a161, plus the current working-tree changes.
- Event-DataCollection source HEAD: 7b2967f72ae27a21cc7ae4c1020992905679d251, plus the current working-tree changes.
- Nested `.gitmodules` files are historical provenance; this flattened snapshot does not contain their gitlinks. Use the restoration manifest instead of assuming recursive submodule checkout.
- Existing component LICENSE and NOTICE files remain authoritative. Frozen Isaac runtime, models, datasets and third-party checkouts are not vendored.

## Current limitations

The source account has exhausted its GPU budget. Migration preparation uses no GPU. The new account cannot access the old filesystem tree; deployment must not depend on source-account absolute paths. REASSEMBLE/DVLA exports mentioned by historical documents are absent from the current source datasets directory and require separate recovery if needed.
