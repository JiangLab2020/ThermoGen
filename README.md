# ThermoGen

ThermoGen is an AI-driven framework for designing hyperthermostable enzymes.
The framework combines generative sequence modeling, protein language model
screening, and structure-guided refinement. Its application in our study
focuses on NAD⁺-dependent methanol dehydrogenases (Mdhs).

This repository provides training and inference code for ThermoGen’s supervised thermostability classifier, a multilayer perceptron that uses mean ESM-2 embeddings to classify protein sequences by thermal adaptation. The training dataset draws on three sources. UniRef50 protein sequences linked to organisms with annotated optimal growth temperatures (OGTs) were used to construct the initial low-temperature (OGT < 30 °C) and high-temperature (OGT ≥ 60 °C) classes. The high-temperature class was expanded with sequences from public MGnify metagenomes and from metagenomic sequencing of geothermal spring samples collected in Tengchong, China. During training, the script reads the numeric suffix of each curated embedding’s `label` field and uses 60 °C as the cutoff to recover the encoded class. At inference, the classifier reports the predicted class and the softmax score for the high-temperature class.


## Installation

Run these commands from the repository root.

```bash
conda env create -f environment.yml
conda activate thermogen
python -m thermogen.train --help
python -m thermogen.predict --help
```

The environment specifies Python 3.12 and PyTorch 2.5.1.
The training and prediction commands use CUDA when available and CPU
otherwise. Pass `--device cpu` to run on CPU explicitly.

## Run the example

### Prepare ESM-2 embeddings

Use the bundled `esm_extract.py` script to generate layer-33 mean embeddings
from `esm2_t33_650M_UR50D`. On first use, the script downloads the pretrained
ESM-2 weights if they are absent from the local cache.
See the [upstream ESM extraction script](https://github.com/facebookresearch/esm/blob/v2.0.0/scripts/extract.py).

```bash
python esm_extract.py esm2_t33_650M_UR50D \
  examples/fasta/candidates.fasta examples/embeddings/candidates \
  --repr_layers 33 --include mean --truncation_seq_length 1022
```

This command truncates sequences longer than 1,022 residues. Each FASTA
header must be unique and contain neither slashes nor backslashes. The
embedding filename is `HEADER.pt`, where `HEADER` is the full text after
`>` with surrounding whitespace removed. The classifier expects a
1,280-element vector in `mean_representations[33]`.


### Predict with the supplied checkpoint

Use a new output directory for each run.

```bash
python -m thermogen.predict \
  --fasta examples/fasta/candidates.fasta \
  --embedding-dir examples/embeddings/candidates \
  --checkpoint checkpoints/best_model_epoch.pt \
  --output-dir examples/candidates_output
```

- `predictions.xlsx` contains one row per input record, with the FASTA header,
  predicted class, and class-1 softmax score.
- `high_tm.fasta` contains the input records assigned to class 1.

You can also pass a compatible classifier checkpoint produced by the training
command below.


## Train

Prepare separate directories of labeled `.pt` embeddings for training and
validation. If starting from FASTA, run the extraction command separately
for each split. Each training and validation FASTA header must end in an
underscore followed by the temperature in °C, such as `>protein_A_63.5`.
ESM stores this header as the embedding's `label` field.

```bash
python -m thermogen.train \
  --train-dir embeddings/train \
  --val-dir embeddings/val \
  --output-dir runs/discriminator
```

The command expects existing, curated data splits. To reproduce the study's
training, use the original curated splits and temperature labels.

The defaults are 100 epochs, a batch size of 1,024, Adam with a learning
rate of 0.0001, StepLR with a 10-epoch step and a multiplier of 0.1, and a
PyTorch seed of 42. The training loader drops its final incomplete batch.
For a smaller training set, choose a batch size between 2 and the number
of training examples.

The output directory must be new. It receives TensorBoard logs, a
`60_tm_discriminator_<epoch>.pt` checkpoint per epoch (starting at 0), and
`best_model_epoch.pt`. The best checkpoint is selected by validation accuracy,
with ties going to the earliest epoch.

Validation uses the full validation split, and loss is averaged by sample
count. ROC AUC and average precision use class-1 scores and are logged when
the split contains both classes.

## License

See [LICENSE](LICENSE) for the repository license.
