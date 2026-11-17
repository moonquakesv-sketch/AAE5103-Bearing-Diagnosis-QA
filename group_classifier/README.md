# Collaborator classification code and data

This contribution was supplied separately by a project teammate. It contains FMD/CWT preprocessing, filename preparation, ViT classifier-head training code, and the full supplied dataset. The three script copies adapt Windows-specific paths to this directory. Function and class definitions are unchanged; exact source/submission hashes and path-only diffs are recorded in [source_manifest.json](source_manifest.json).

## Restore the complete data

From the repository root:

```bash
python group_classifier/prepare_data.py
python group_classifier/prepare_data.py --verify-only
```

The checked-in `dataset/` volumes contain **all 1160 supplied data files**, with an ordered-volume manifest and SHA-256 for every original file. The helper uses only the Python standard library, checks the archive and extracted files, and restores the original `Data/` layout. It does not download anything, train a model or overwrite a different existing file. Allow disk space for the extracted data and a temporary assembled archive.

| Data directory | Files | Contents |
| --- | ---: | --- |
| `Data/HP_0`–`Data/HP_3` | 48 | Supplied XLSX vibration records; 12 at each load |
| `Data/train_1s` | 822 | Four-class processed training arrays |
| `Data/test_1s` | 290 | Four-class processed arrays used as validation and final evaluation by the supplied script |

All 1112 NPY arrays were checked as finite `float32` arrays with shape `(3, 224, 224)`. XLSX containers were checked without executing macros or the scripts. This packaging check does not independently establish upstream record selection, specimen independence or model accuracy.

## Supplied scripts

1. [1_whole pre-processing.py](code/1_whole%20pre-processing.py): spreadsheet loading, high-pass filtering, FMD, CWT and resizing. Executable defaults use `fs=12000`, `win=12000`, `step=6000` (one-second windows with 50% overlap), despite an older half-second description in its introductory comment. Its default file list processes HP_0.
2. [2_rename.py](code/2_rename.py): rename the generated segments using fault and load prefixes. Original rename/copy behavior is retained.
3. [3_Fine-tuning model_vit.py](code/3_Fine-tuning%20model_vit.py): frozen ViT-B/16 backbone with a trained four-class classification head, labels `B`, `IR`, `N`, `OR`. It uses `test_1s` during checkpoint selection/early stopping and later reports on the same set; that set is not an untouched final test.

Runtime imports are listed in [requirements-reference.txt](requirements-reference.txt). The teammate's exact tested environment was not supplied. Use its compatible TensorFlow, TensorFlow Addons and `vit-keras` environment separately from the PyTorch Q&A demonstration; this work did not install those libraries or execute/retrain the ViT script.

No trained `.h5` checkpoint, saved ViT predictions, or ResNet-50 implementation was present in the supplied package. The ViT source and data are included for review and reproduction. They are not connected to the repository's ResNet-18/SVM Q&A demo, and its saved D04 or E02 results do not describe this ViT model.

The supplied filenames/code identify the data as CWRU-derived. Per-file conversion from original MAT records to these XLSX/NPY files was not independently reconstructed here. The full provided data and their hashes are retained without changing their values or train/test placement.
