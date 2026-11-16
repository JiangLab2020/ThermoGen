# ThermoGen

ThermoGen is an AI-driven framework for designing hyperthermostable enzymes.
The framework combines generative sequence modeling, protein language model
screening, and structure-guided refinement. Its application in our study
focuses on NAD⁺-dependent methanol dehydrogenases (Mdhs).

This code release contains the supervised thermostability screening component.
It classifies the mean ESM-2 embedding of a protein sequence into class 0
(training temperature label < 60 °C) or class 1 (training temperature label
≥ 60 °C). The notebook derives this label from the numeric suffix of each
embedding's `label` field. The score is the classifier's class-1 softmax output.

## Installation

Run these commands from the repository root.

```bash
conda env create -f environment.yml
conda activate thermogen
python -m thermogen.train --help
python -m thermogen.predict --help
```

The release environment uses Python 3.12 and PyTorch 2.5.1. The supplied
research environment specified Python 3.7, PyTorch 1.10.1, and CUDA 11.1.
Both commands default to CUDA when available and otherwise use CPU.
Use `--device cpu` to select CPU explicitly.

## Prepare ESM-2 embeddings

Generate the layer-33 mean representation with
[the official ESM v2.0.0 extraction script](https://github.com/facebookresearch/esm/blob/v2.0.0/scripts/extract.py).
On first use, the script downloads the pretrained ESM-2 weights if they are absent
from the local cache.

```bash
curl -fLo esm_extract.py https://raw.githubusercontent.com/facebookresearch/esm/v2.0.0/scripts/extract.py
python esm_extract.py esm2_t33_650M_UR50D candidates.fasta embeddings/candidates \
  --repr_layers 33 --include mean --truncation_seq_length 1022
```

The command above truncates sequences longer than 1,022 residues. For inference,
each `>HEADER` must be unique and map to `HEADER.pt`, using the full FASTA
header text after `>` with surrounding whitespace trimmed. Headers must contain no slash or backslash. For
training, use headers ending in the temperature in °C, such as
`>protein_A_63.5`, so that ESM stores the label with the embedding.
The classifier expects 1,280 values in `mean_representations[33]`.

## Train

Prepare separate directories of labeled `.pt` files for training and validation.
If starting from FASTA, run the extraction command separately for each
existing split. To reproduce the manuscript's split and label curation,
use the original curated training and validation sets. This release expects
the output of label curation as input.

```bash
python -m thermogen.train \
  --train-dir embeddings/1_train \
  --val-dir embeddings/1_val \
  --output-dir runs/discriminator
```

The default configuration uses 100 epochs, batch size 1,024, Adam at
0.0001, and StepLR with a 10-epoch step and a multiplier of 0.1.
For a smaller training set, pass a batch size between 2 and the number of
training examples. The default PyTorch random seed is 42.

The output directory must be new. It receives TensorBoard logs, one
`60_tm_discriminator_<epoch>.pt` checkpoint per epoch (starting at 0),
and `best_model_epoch.pt`, selected by maximum validation accuracy.
The first checkpoint wins a tie.

The release computes metrics on the full validation split, averages loss
by sample count, and computes ROC AUC and average precision from class-1
scores.

## Predict

Pass the original classifier's state dictionary or a checkpoint generated
by the training command. Use a new directory for prediction output.

```bash
python -m thermogen.predict \
  --fasta examples/fasta/candidates.fasta \
  --embedding-dir examples/embeddings/candidates \
  --checkpoint checkpoints/best_model_epoch.pt \
  --output-dir examples/candidates_output
```

- `predictions.xlsx` contains one row per FASTA record, with its identifier,
  predicted class, and class-1 score.
- `high_tm.fasta` contains all records predicted as class 1.

The manuscript's curated splits, classifier weights, and sequence design
and structural refinement workflows are separate research assets.
