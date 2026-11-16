"""Score FASTA sequences using extracted ESM-2 embeddings and a checkpoint."""

import argparse
from pathlib import Path

import torch
from Bio import SeqIO
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell

from .model import Thermo, load_embedding, resolve_device


@torch.inference_mode()
def write_predictions(records, model, args, device):
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("Predictions")
    sheet.append(["Sequence_ID", "Thermostability", "Positive_score"])
    positive_count = 0

    with (args.output_dir / "high_tm.fasta").open("w") as fasta:
        for offset in range(0, len(records), args.batch_size):
            batch = records[offset:offset + args.batch_size]
            vectors = [load_embedding(args.embedding_dir / f"{record.description}.pt")[1] for record in batch]
            logits = model(torch.stack(vectors).to(device))
            labels = logits.argmax(dim=1).tolist()
            scores = logits.softmax(dim=1)[:, 1].tolist()
            for record, label, score in zip(batch, labels, scores):
                identifier = WriteOnlyCell(sheet, value=record.description)
                identifier.data_type = "s"
                sheet.append([identifier, label, score])
                if label == 1:
                    SeqIO.write(record, fasta, "fasta-2line")
                    positive_count += 1

    workbook.save(args.output_dir / "predictions.xlsx")
    print(f"Scored {len(records)} sequences. Class 1: {positive_count}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")

    records = list(SeqIO.parse(args.fasta, "fasta"))
    if not records:
        parser.error(f"No sequences found in {args.fasta}")
    seen = set()
    for record in records:
        record.description = record.description.strip()
        identifier = record.description
        if not identifier or identifier in {".", ".."} or any(c in identifier for c in "/\\"):
            parser.error(f"Invalid FASTA identifier: {identifier!r}")
        if identifier in seen:
            parser.error(f"Duplicate FASTA identifier: {identifier}")
        seen.add(identifier)
        path = args.embedding_dir / f"{identifier}.pt"
        if not path.is_file():
            parser.error(f"Missing embedding: {path}")

    device = resolve_device(args.device)
    model = Thermo()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_predictions(records, model, args, device)


if __name__ == "__main__":
    main()
