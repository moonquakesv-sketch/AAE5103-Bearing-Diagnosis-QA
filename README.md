# Bearing Fault Diagnosis and Evidence-Based Maintenance Q&A

AAE5103 course project with two separately supplied components: (1) a working ResNet-18/Scaler/SVM, graph-retrieval and maintenance-Q&A demonstration; (2) a collaborator's FMD/CWT and ViT classification source with its complete supplied dataset. The collaborator classifier has not been connected to the Q&A demo.

## Collaborator code and complete data

See [group_classifier/](group_classifier/README.md) for the three supplied classification scripts and all 1160 supplied data files (48 XLSX records and 1112 processed arrays). Data are stored in checked-in compressed volumes. Restore the exact original files locally with:

```bash
python group_classifier/prepare_data.py
```

Every archive volume and restored file is checked against its SHA-256. The collaborator TensorFlow/ViT environment is separate from the PyTorch demo environment below. No trained collaborator checkpoint or saved predictions were provided.

## Q&A demo quick start

Use Python 3.12 on macOS or Linux. The current implementation uses `fcntl`; native Windows has not been validated.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B app.py --check
python -B app.py
```

Open `http://127.0.0.1:7862/`. Choose either demonstration case and click **Replay Saved Record**. Replay needs no API key or pretrained-weight download and sends no new model requests. English translations of historical Chinese answers are labelled in the interface.

For fresh local image classification, first fetch the original fixed weight:

```bash
python scripts/fetch_submission_assets.py
python -B app.py --check --require-assets
```

To use **Run Live Demo**, configure your own `DEEPSEEK_API_KEY` environment variable before starting the application. Each manual run performs local classification, task identification, graph retrieval, and answer generation, with at most two provider requests and no automatic retries. Without a key, saved replay remains available. The configured historical model alias is `deepseek-flash`; current provider availability and live output have not been retested for this distribution.

`Start Demo.command` is a convenience launcher for the active Python environment on macOS. Use `python -B app.py --port 7865` to choose another local port.

## Inspect the saved results

```bash
python -B scripts/recompute_submission_results.py
python -B -m unittest discover -s runs/i01/20261106_200240_english_demo_rev053_287f5f2b/tests -v
python -B -m unittest discover -s tests -p 'test_submission_portability.py' -v
```

The recomputation reads existing predictions and point scores. It performs no new classifier inference, answer generation or model grading.

| Material | Location |
| --- | --- |
| Collaborator source and complete data | [group classifier](group_classifier/README.md) |
| Current demo | [English demo implementation](runs/i01/20261106_200240_english_demo_rev053_287f5f2b/) |
| Data processing, classifier and Q&A modules | [scripts/](scripts/) |
| Frozen classifier and training/validation evidence | [D03 formal model](runs/d03/20261022_rev006/) |
| Knowledge tables and source excerpt provenance | [knowledge/i01/v1/](knowledge/i01/v1/) |
| Original record and window mappings | [records.csv](data/records.csv), [samples.csv](data/samples.csv) |
| Sole formal classification test | [D04 metrics](runs/d04/20261023_rev008/metrics.json), [236 saved predictions](runs/d04/20261023_rev008/test_predictions.csv) |
| Five-group REF results | [statistics](runs/e02_ref4_acceptance/20261106_024041_rev050_ref4/reference_statistics.json), [contrasts](runs/e02_ref4_acceptance/20261106_024041_rev050_ref4/contrasts.json), [figure](runs/e02_ref4_acceptance/20261106_024041_rev050_ref4/figures/REF_G_vs_L_and_T.png) |
| Questions, rubrics and exposure records | [evaluation/e01/](evaluation/e01/) |
| Distribution changes and verification scope | [submission notes](docs/SUBMISSION_NOTES.md) |

## Data and source material

The Q&A demonstration includes four bound training/validation sample images and its fitted classification artifacts. Its bulk original MAT data, batch time-frequency images, pretrained weights, native tokenizer libraries and complete third-party webpages are acquired separately. The collaborator's separately supplied XLSX/NPY dataset is included in full as archive volumes under `group_classifier/dataset/`.

- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome) is the source of the selected vibration records. Exact record URLs and hashes are retained in `data/records.csv`. `python scripts/fetch_submission_assets.py --raw-data` fetches the 16 selected MAT files. Frozen preprocessing is documented in `configs/data.yaml` and `scripts/d02_pipeline.py`; regeneration is separate from viewing the demo and may require the original workflow's frozen audit inputs.
- The official ResNet-18 IMAGENET1K_V1 weight is downloaded by the asset helper and checked against its original SHA-256. The fitted SVM pipeline is included. Pretrained weights have their own usage terms; see [TorchVision's guidance](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights).
- The distribution contains the existing short source quotations and authored knowledge statements, with a pinned excerpt index. The six full NSK webpage snapshots stay outside the repository. At packaging, the excerpts were checked against the hash-verified local snapshots; the offline loader checks exported excerpt identity. It does not perform a new online source verification.
- `python scripts/fetch_submission_assets.py --historical-sources` optionally obtains the old full snapshots for legacy source audits. A changed webpage is rejected if its bytes do not match the historical hash. These optional snapshots remain ignored by Git.

See [third-party notices](THIRD_PARTY_NOTICES.md) and [external asset identities](external_assets.json).

## Evaluation limits

The saved D04 accuracy, macro-F1 and balanced accuracy are 1.0000 on 236 windows from the selected four CWRU test records at the held-out load. Windows are not independent bearing specimens. The result does not establish generalization to unseen specimens or aviation equipment; that test set has already been used.

The five-group REF evaluation retains all 65 planned outputs: 62 final scores are valid and three remain unknown. Its scores are model judged. The debug G−L coverage difference ranges from −1.6071 to +12.6786 percentage points because of unknown scores; this range is not a confidence interval. G−L coverage on the two operational-reserve questions is −82.1429 points. These results do not establish overall graph superiority. G−T's debug advantage partly reflects a historical service failure.

An image class selects a retrieval route. The associated scenario is a teaching assumption. Citation-ID checks do not establish answer correctness. This is a limited course prototype.

## Historical implementation

Earlier source, configuration and consumed-run results are included for review. Their original whole-workspace seals, platform-specific tokenizer registrations and one-time paid execution gates belong to the historical runs. The current entry points are `app.py` and the saved-result recomputation script above. Existing batch launchers direct to the read-only recomputation; historical authorizations and consumed identities are not renewed.

Original model freezes, numerical artifacts, knowledge tables, raw saved answers and canonical scores retain their original bytes. Some historical records contain the source machine's path as provenance metadata; the current entry points resolve their own paths from the repository location.
