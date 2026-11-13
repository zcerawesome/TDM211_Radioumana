# Source and derived-data license

The README accompanying the local float32 release is retained (trailing
whitespace normalized) as
[SOURCE_DATASET_README.txt](SOURCE_DATASET_README.txt). It identifies **Spectrum
Highway Dataset 2**, Fraunhofer Institute for Integrated Circuits IIS, and states
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**.
Its source repository is [FIOT Highway2](https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/fiot_highway2).
The live source page could not be fetched during packaging; the bundled release
README is the evidence for this dataset version's license, not an assertion
about a different future release.

License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
([legal text](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)).

## Attribution and changes

Credit Fraunhofer IIS and the source publication:
Lucas Heublein, Nisha L. Raichur, Tobias Feigl, Tobias Brieger, Fin Heuer,
Lennart Asbach, Alexander Rügamer and Felix Ott,
“Evaluation of (Un-)Supervised Machine Learning Methods for GNSS Interference
Classification with Real-World Data Discrepancies,” ION GNSS+ 2024,
pp. 1260–1293, [DOI: 10.33012/2024.19887](https://doi.org/10.33012/2024.19887).
Further citations, acknowledgments and the source license notice are preserved
in SOURCE_DATASET_README.txt.

The Aerospace Corporation converted the distributed arrays to float32 on
2026-08-27, as noted in that README. This Purdue Data Mine / Aerospace Task 1
work adds small receiver-gain and nonnegative noise-power perturbations to
training spectrograms, retaining class labels. It creates derived comparison
plots, class summaries and evaluation results. No endorsement by the original
creators is implied.

The derived synthetic arrays and plots distributed with this Task 1 work are
provided under **CC BY-NC-SA 4.0**. Other original content in this results folder
is also offered under CC BY-NC-SA 4.0 to the extent copyright applies. Repository
Python code remains under the existing root [LICENSE](../../LICENSE); this
notice does not relicense the dataset as software.

## Sharing conditions

Keep this notice, source attribution, source/license links and the description
of changes with any shared archive or storage folder. Retain the source README
and license notices. Share adapted data under the same license; do not impose
additional legal or technological restrictions that remove rights the license
grants. Use must be noncommercial. Corporate sponsorship by itself does not
grant permission for commercial use; a use primarily intended for commercial
advantage or monetary compensation needs separate permission from the rights
holder if it falls outside this license.

This package shares research code, compact results and properly attributed
plots. It does not upload original data or synthetic arrays. Before any later
data upload, include these notices in the artifact itself and ensure the
intended use fits the noncommercial license. Storage access controls are for
team administration, not a substitute license or a restriction on recipients'
licensed rights.
