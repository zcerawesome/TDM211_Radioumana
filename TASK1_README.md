# Task 1 — Highway2 synthetic training data

**Completed and verified:** 11,865 original fitting samples + **1,359 synthetic
samples** = **13,224 training samples**. Classes 3–8 were augmented. The existing
SqueezeNet 1.1 baseline successfully completed five optimizer updates using
both original and synthetic samples, plus a validation batch. The original
18-test suite passed; **20 tests passed** after adding two collaboration/portability tests.
Task 2 is not part of this change.

Read the [research report](experiments/highway2_task1/REPORT.md),
[metrics and charts](experiments/highway2_task1/results/),
[execution evidence](experiments/highway2_task1/evidence/), and
[data attribution/license](experiments/highway2_task1/DATA_LICENSE.md).

## Setup and reproduce

Use Python >=3.11. Commands below are for a POSIX shell (Linux/macOS; on Windows
use WSL or equivalent environment-variable commands). Clone the repository,
check out the feature branch while the PR is under review, and install:

```bash
git clone https://github.com/zcerawesome/TDM211_Radioumana.git
cd TDM211_Radioumana
git switch feature/highway2-synthetic-augmentation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Obtain the **float32 Highway2 release** from the team's authorized source. Its
root must contain `train.txt`, `test.txt`, `README.md`, and `data/*.npy`. The
verified release has 16,480 arrays. Use your own local paths; no author's Mac
paths or credentials are needed. Avoid spaces in dataset/output paths because
the upstream two-column manifest format uses whitespace separators.

```bash
export HIGHWAY2_ROOT=/path/to/fiot_highway2_f32
export TASK1_OUTPUT="$PWD/artifacts/highway2_task1"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

python -m radiomana.highway2_synthetic \
  --root "$HIGHWAY2_ROOT" --out "$TASK1_OUTPUT" \
  --seed 20261112 --ratio 0.1 --max-variants 2
python -m pytest tests -q
python examples/validate_task1.py --root "$TASK1_OUTPUT"
python examples/task1_neural_smoke.py --root "$TASK1_OUTPUT" \
  --result "$TASK1_OUTPUT/neural_integration.json"
# Optional: reproduce the inexpensive original-vs-augmented CPU experiment.
python examples/task1_baseline.py --root "$TASK1_OUTPUT"
```

Generation refuses an existing output directory and writes outside the original
dataset. Keep a successful output for subsequent training; do not regenerate it
on every run. For repeated neural smoke tests use a new `--result` filename.
`validate_task1.py` reproduces variants in memory without rewriting arrays.
No GPU is required. Generation streams one array at a time. The CPU baseline
caches 125 scalar features per sample. The neural smoke test uses two CPU threads,
float32 and batches of three. Exact bitwise reproduction was verified with the
versions recorded in [environment.json](experiments/highway2_task1/results/environment.json);
other numerical-library versions may require investigation if hashes differ.

## Load for model training

The following uses the environment variables set above:

```python
import os
from radiomana.synthetic_data import SyntheticHighwayDataModule
from radiomana.models import HighwayBaselineModel
import lightning as L

data = SyntheticHighwayDataModule(os.environ["TASK1_OUTPUT"], augmented=True,
                                  batch_size=3, num_workers=0)
data.setup()
assert len(data.data_train) == 13224
x, y = next(iter(data.train_dataloader()))  # float32 [3,512,243], int64 labels

model = HighwayBaselineModel(num_classes=9)
trainer = L.Trainer(accelerator="cpu", devices=1, precision="32-true",
                    max_epochs=1, limit_train_batches=5, limit_val_batches=1,
                    num_sanity_val_steps=0, logger=False,
                    enable_checkpointing=False)
trainer.fit(model, datamodule=data)
```

Use `augmented=False` for the original-fitting-only control with the same held-out
data. The dedicated smoke command deliberately samples nine originals (all nine
classes) and six synthetics (classes 3–8), checks actual forward-pass inputs,
finite losses/gradients and parameter updates; random batches in the example
above do not guarantee rare-class coverage in only five steps.

**Do not use the ordinary `HighwayDataModule` on the generated `train.txt`.** It
would split augmented data again and could separate a parent from its variants.
`SyntheticHighwayDataModule` reuses RadioManA loading/DataLoaders while honoring
the saved train/validation/test manifests. Original references are absolute
paths to *your* local dataset; synthetic references are relative to the output.
If moving an existing result folder, use the relocation command below rather
than editing or re-splitting the manifests manually.

## Method and verified results

Freeze the repository's stratified 90/10 split with seed `0xD00D1E` before any
augmentation. The original test set (3,296 samples) and validation set (1,319)
are never augmentation sources. Select any fitting class below
`ceil(0.1 * 4997) = 500` samples, including background class 3. Generate toward
500, capped at two variants per original. Counts are chosen from training only.

| Label | Class | Original fitting | Synthetic | Augmented |
|---|---|---:|---:|---:|
| 0 | Background a | 3,773 | 0 | 3,773 |
| 1 | Background b | 2,074 | 0 | 2,074 |
| 2 | Background c | 4,997 | 0 | 4,997 |
| 3 | Background d | 43 | 86 | 129 |
| 4 | Chirp, high distance | 158 | 316 | 474 |
| 5 | Chirp, medium distance | 259 | 241 | 500 |
| 6 | Chirp, small distance | 302 | 198 | 500 |
| 7 | Cigarette lighter 1 | 144 | 288 | 432 |
| 8 | Cigarette lighter 2 | 115 | 230 | 345 |

Data are `[512,243]` log-power spectrograms, not raw IQ. Apply receiver power
gain uniformly in ±0.25 dB and add nonnegative gamma-distributed noise power
(shape 16), 14–20 dB below the source's first percentile. Stable log-add-exp
adds **linear powers**, avoiding the repository's signed-noise clipping issue.
The gamma averaging factor is an augmentation assumption, not known metadata.
No flips, rotations, frequency shifts, time wrapping/cropping, resizing or
jammer relabeling are performed. Labels and native float32 shape are preserved.

Seed 20261112 controls parent selection and recorded per-sample seeds. Quality
gates require correlation >=0.995, RMSE <=1 dB and total-power change <=0.3 dB.
All 1,359 samples loaded and reproduced exactly; no strongest-frequency-bin
changes were observed. Exact duplicate detection and source integrity hashes
passed. The existing SqueezeNet/AdamW Lightning pipeline completed **five
optimizer updates**, with actual model inputs containing **9 originals and 6
synthetics**, and one validation batch. No integration bug required changes to
the production model or augmentation implementation.

The fixed CPU logistic-regression baseline used the same untouched test set:
macro-F1 **0.6303 → 0.6484**, balanced accuracy **60.78% → 65.11%**, overall
accuracy **77.82% → 77.37%**. This single experiment is not a full neural-training
comparison, significance test or a control isolating augmentation from class
reweighting. Synthetic variants are correlated with parents, not independent
new captures. No session IDs/timestamps are available; exact content disjointness
cannot establish session independence. CPU smoke coverage does not certify
MPS/CUDA or model generalization.

## Shared storage and portable reuse

Measured synthetic size: **676,499,328 bytes / 645.16 MiB in 1,359 files**. Although
each file is small, the aggregate is unsuitable for routinely versioning in Git.
GitHub [recommends small repositories](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Use the team's approved **Purdue project storage (for example, its existing
Anvil allocation)** or institutional shared Drive/OneDrive for a versioned result
folder/archive. Git LFS or release assets are alternatives only if the team
accepts their storage/usage policies; no external data upload was performed here.

Share the generated folder with its synthetic arrays, manifests, summary,
provenance, integrity CSV and `COMPLETE`, plus `DATA_LICENSE.md` and
`SOURCE_DATASET_README.txt` from this repository. Preserve checksums and record a
version and download location in the team's storage index. The original dataset
should be obtained separately. The [CC BY-NC-SA 4.0 conditions](experiments/highway2_task1/DATA_LICENSE.md)
apply to derivative arrays and plots: preserve attribution and license links,
identify changes, share alike and use noncommercially. Corporate sponsorship
does not automatically authorize commercial use.

After downloading a complete shared result folder, rebind its metadata into a
new local view without regenerating or copying the arrays:

```bash
python examples/task1_relocate.py --input /path/to/downloaded_result \
  --data-root "$HIGHWAY2_ROOT" --out "$TASK1_OUTPUT"
python examples/validate_task1.py --root "$TASK1_OUTPUT"
```

Relocation requires matching source manifests, preserves split membership/order
and labels, and symlinks the downloaded `synthetic/` directory. Keep that download
in place. It updates only copied metadata and path references. Run the neural
smoke command with a fresh result filename to verify the relocated setup.

## Repository layout

- `radiomana/highway2_synthetic.py`: streaming audit and augmentation pipeline.
- `radiomana/synthetic_data.py`: frozen-manifest dataset and Lightning datamodule.
- `examples/task1_*.py`, `examples/validate_task1.py`: evaluation, neural smoke,
  portable metadata relocation and exhaustive validation.
- `tests/test_highway2_synthetic.py`: regression and CLI integration tests.
- `experiments/highway2_task1/REPORT.md`: research findings and limitations.
- `experiments/highway2_task1/results/`: compact charts, CSVs, metrics, provenance
  and integrity evidence. These are historical results, not a runnable dataset.
- `experiments/highway2_task1/evidence/`: execution logs with machine paths removed.

Runtime arrays, environments, caches and generated outputs are ignored by Git.
No original or synthetic `.npy` arrays are included in this branch.
