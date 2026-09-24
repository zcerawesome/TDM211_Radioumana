# Radio Mana

*radiomana* is an open-source PyTorch library developed by *The Aerospace Corporation* for **GNSS jamming detection and classification** using deep learning on radio frequency (RF) spectrum data. This library is constructed to support our University Partnership Program (UPP) with Purdue University and to provide students and researchers with a framework for developing and testing neural network models for jamming detection.

## What it does

This library provides:
- **Data loading utilities** for the Fraunhofer GNSS Jamming Highway2 Dataset
- **Neural network models** (HighwayBaselineModel) for jamming classification
- **Training pipelines** using PyTorch Lightning for jamming detection research
- **Signal processing transforms** (noise augmentation, time cropping) optimized for RF data

## Quick Start

1. Install

    ```bash
    # install in development mode
    pip install --editable .
    ```
2. Download the [Spectrum Highway Dataset 2](https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/fiot_highway2)
3. Set environment variable in your `.bashrc` file:
   ```bash
   export DSET_FIOT_HIGHWAY2=/path/to/highway2/dataset
   ```

## Usage Examples

### Basic Data Loading

```python
import radiomana

# inspect a sigle sample and label
dataset = radiomana.Highway2Dataset()
psd_sample, label = dataset[0]  # power spectral density + jamming classification

# inspect a training batch
datamodule = radiomana.HighwayDataModule(batch_size=32, num_workers=4)
datamodule.setup()
train_batch = next(iter(datamodule.train_dataloader()))
```

### Model Inference

```python
# create and train your own model
model = radiomana.HighwayBaselineModel(num_classes=9)

# or load from checkpoint after training
model.load_state_dict(torch.load("path/to/your/trained_model.pt"))

# classify RF spectrum
with torch.no_grad():
    logits = model(psd_sample.unsqueeze(0))
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    confidence = probabilities[0, predicted_class]
```

### Training Your Own Model

```bash
# train model
python examples/train_baseline.py

# benchmark model inference speed
python examples/bench_model.py
```

## Model Performance

Performance on the Highway2 GNSS jamming detection dataset using the provided HighwayBaselineModel.

| Model                |  Submodel  | Augmentations | Params (M) | Memory (MB) | multadds (G) | Test Loss | F1    | Acc% |
|----------------------|------------|---------------|------------|-------------|--------------|-----------|-------|------|
| HighwayBaselineModel | squeezenet | None          | 0.7        | 3           | 0.6          | 0.618     | 0.512 | 76.0 |

## Dataset Classes

Highway2 has 9 classes: labels 0–3 are "no interference" (background variants) and labels 4–8 are interference (chirp at high/medium/small distance, and two cigarette-lighter jammer variants).

Highway1 (11 classes) uses a different taxonomy and doesn't share interference classes 1:1 with Highway2, so for cross-dataset validation both datasets can be collapsed to binary `not_jammed` / `jammed`:

|                  | Highway1       | Highway2          |
|------------------|----------------|-------------------|
| `not_jammed` (0) | labels 0, 1, 2 | labels 0, 1, 2, 3 |
| `jammed` (1)     | labels 3–10    | labels 4–8        |

## Open Source Details

### Release

This project is approved for public release with unlimited distribution by Aerospace under OSS Project Ref #OSS25-0006.

### Contributing

Do you have code you would like to contribute to this Aerospace project?

We are excited to work with you. We are able to accept small changes
immediately and require a Contributor License Agreement (CLA) for larger
changesets. Generally documentation and other minor changes less than 10 lines
do not require a CLA. The Aerospace Corporation CLA is based on the well-known
[Harmony Agreements CLA](http://harmonyagreements.org/) created by Canonical,
and protects the rights of The Aerospace Corporation, our customers, and you as
the contributor. [You can find our CLA here](https://aerospace.org/sites/default/files/2020-12/Aerospace-CLA-2020final.pdf).

Please complete the CLA and send us the executed copy. Once a CLA is on file we
can accept pull requests on GitHub or GitLab. If you have any questions, please
e-mail us at [open-source@aero.org](mailto:open-source@aero.org).

### Licensing

The Aerospace Corporation supports Free & Open Source Software and we publish
our work with GPL-compatible licenses. If the license attached to the project
is not suitable for your needs, our projects are also available under an
alternative license. An alternative license can allow you to create proprietary
applications around Aerospace products without being required to meet the
obligations of the GPL. To inquire about an alternative license, please get in
touch with us at [open-source@aero.org](mailto:open-source@aero.org).
# TDM211_Radioumana
