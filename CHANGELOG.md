# Changelog

## v1.0.0 — Final release

### Fixed
- Test set was loaded from `afhq/train`, so "test" accuracy was measured on training images. It now uses the held-out `afhq/val` split.
- `ReduceLROnPlateau` monitored `val_acc`, a metric that does not exist, so the learning rate was never reduced. It now monitors `val_loss`.
- Training kept the weights from the last epoch (validation accuracy dropped to 0.69 in the final epoch). Early stopping now restores the best epoch.
- Prediction labels were a mix of strings and integers after remapping, which marked some correct predictions as wrong. Labels are now class names end to end.

### Added
- Classification report, confusion matrix and training curves.
- Misclassified-image gallery with prediction confidence.
- Stratified train/validation split and fixed random seeds.
- Best model saved as `afhq_cnn.keras`.
- Runs locally as well as on Kaggle (paths auto-detected, overridable via `AFHQ_DIR` / `OUT_DIR`).
- `requirements.txt`, `LICENSE`, `.gitignore`.

### Changed
- Gentler augmentation (rotation 45° → 15°) suited to upright face crops.
- Notebook cleaned up: duplicate cells removed, configuration in one place.
- README rewritten to describe the actual 3-class task, data splits and how to run.
