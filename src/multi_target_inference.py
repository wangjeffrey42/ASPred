#!/usr/bin/env python3
"""
multi_target_inference.py
Run all antigen models against a single input CSV and combine results into one output.
"""

import argparse
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import transformers
transformers.logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()

from safetensors.torch import load_file as st_load
from peft import PeftModel, PeftConfig
from transformers import EsmTokenizer, EsmForSequenceClassification


def load_model(model_path, device):
    cfg = PeftConfig.from_pretrained(model_path)
    base_name = cfg.base_model_name_or_path
    base = EsmForSequenceClassification.from_pretrained(base_name, num_labels=2)
    model = PeftModel.from_pretrained(base, model_path)

    # Inject classifier weights (PEFT version mismatch fix)
    sd = st_load(os.path.join(model_path, "adapter_model.safetensors"))
    classifier_state = {
        k.replace("base_model.model.", ""): v
        for k, v in sd.items()
        if "classifier" in k
    }
    if classifier_state:
        model.base_model.model.load_state_dict(classifier_state, strict=False)

    return model.to(device).eval(), EsmTokenizer.from_pretrained(base_name)


def predict(model, tokenizer, sequences, threshold=0.5, batch_size=32):
    all_probs, all_preds = [], []
    device = next(model.parameters()).device
    sequences = [s for s in sequences if isinstance(s, str) and len(s) > 0]

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**tokens).logits, dim=1)
            preds = (probs[:, 1] > threshold).int()
        all_probs.extend(probs[:, 1].cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return all_probs, all_preds


def pick_seq_column(df, override=None):
    if override:
        if override in df.columns:
            return override
        raise KeyError(f"--seq_column '{override}' not found. Available: {list(df.columns)}")
    for col in ("sequence_alignment_aa", "sequence_aa", "sequence"):
        if col in df.columns:
            return col
    raise KeyError("No sequence column found. Tried: sequence_alignment_aa, sequence_aa, sequence")


def discover_models(base_dir, objective):
    models_dir = os.path.join(base_dir, "models")
    found = {}
    if not os.path.isdir(models_dir):
        return found
    for entry in sorted(os.listdir(models_dir)):
        parts = entry.split("_")
        if len(parts) >= 4 and parts[-1] == objective:
            antigen = parts[-2].upper()
            adapter_path = os.path.join(models_dir, entry, "training_output")
            if os.path.isfile(os.path.join(adapter_path, "adapter_model.safetensors")):
                found[antigen] = adapter_path
    return found


def main():
    ap = argparse.ArgumentParser(
        description="Multi-target ASPred — run all antigen models at once and combine results"
    )
    ap.add_argument("--input_csv",  required=True, help="CSV file with antibody sequences")
    ap.add_argument("--base_dir",   default=".",   help="Root of ASPred repo (contains models/)")
    ap.add_argument("--objective",  default="f1",  choices=["f1", "loss"],
                    help="Which model variant to use (f1 or loss)")
    ap.add_argument("--threshold",  type=float, default=0.5)
    ap.add_argument("--batch_size", type=int,   default=32)
    ap.add_argument("--output_dir", default="results/multi_target")
    ap.add_argument("--seq_column", default=None,
                    help="Sequence column name (auto-detected if omitted)")
    ap.add_argument("--antigens", nargs="+", default=None,
                    help="Subset of antigens to run e.g. --antigens FLU SARS (default: all found)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Discover available models
    available = discover_models(args.base_dir, args.objective)
    if not available:
        raise RuntimeError(
            f"No models found in {args.base_dir}/models/ with objective '{args.objective}'"
        )

    targets = [t.upper() for t in args.antigens] if args.antigens else sorted(available.keys())
    missing = [t for t in targets if t not in available]
    if missing:
        raise RuntimeError(f"No model found for: {missing}. Available: {sorted(available.keys())}")

    print(f"Antigens: {targets}")
    print(f"Models:   {args.objective}-optimized")

    # Load sequences
    df = pd.read_csv(args.input_csv)
    seq_col = pick_seq_column(df, args.seq_column)
    sequences = df[seq_col].tolist()
    print(f"Input:    {len(sequences)} sequences from column '{seq_col}'\n")

    # Run each model and attach results
    results = df.copy()
    for antigen in targets:
        model_path = available[antigen]
        print(f"=== {antigen} ===")
        print(f"  Model: {model_path}")
        model, tokenizer = load_model(model_path, device)
        probs, preds = predict(model, tokenizer, sequences,
                               threshold=args.threshold, batch_size=args.batch_size)
        results[f"{antigen}_prob"] = probs
        results[f"{antigen}_pred"] = preds
        pos = sum(preds)
        print(f"  Positive: {pos}/{len(preds)} ({100*pos/len(preds):.1f}%)\n")
        del model  # free memory before loading next model

    # Add summary columns
    pred_cols = [f"{a}_pred" for a in targets]
    results["n_targets_positive"] = results[pred_cols].sum(axis=1)
    results["any_positive"] = (results["n_targets_positive"] > 0).astype(int)

    # Save combined CSV
    out_csv = os.path.join(args.output_dir, "multi_target_predictions.csv")
    results.to_csv(out_csv, index=False)
    print(f"Saved combined predictions: {out_csv}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Antigen':<10} {'Positive':>10} {'Negative':>10} {'% Positive':>12}")
    print("-" * 46)
    for antigen in targets:
        pos = int(results[f"{antigen}_pred"].sum())
        neg = len(results) - pos
        print(f"{antigen:<10} {pos:>10} {neg:>10} {100*pos/len(results):>11.1f}%")
    print("-" * 46)
    multi = int((results["n_targets_positive"] > 1).sum())
    any_pos = int(results["any_positive"].sum())
    print(f"{'Any antigen':<10} {any_pos:>10} {len(results)-any_pos:>10} {100*any_pos/len(results):>11.1f}%")
    print(f"{'Multi-target':<10} {multi:>10}   (positive for 2+ antigens)")

    # Save FASTA of sequences positive for any antigen
    pos_df = results[results["any_positive"] == 1]
    fasta_path = os.path.join(args.output_dir, "multi_target_positives.fasta")
    with open(fasta_path, "w") as f:
        for i, row in pos_df.iterrows():
            seq_id = row.get("sequence_id", f"seq{i}")
            tags = "+".join([a for a in targets if row[f"{a}_pred"] == 1])
            f.write(f">{seq_id} targets={tags}\n{row[seq_col]}\n")
    print(f"\nSaved positives FASTA: {fasta_path}")

    # Plot 1: probability distributions per antigen
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), squeeze=False)
    for i, antigen in enumerate(targets):
        ax = axes[0][i]
        sns.histplot(results[f"{antigen}_prob"], kde=True, bins=30, ax=ax, color="steelblue")
        ax.axvline(args.threshold, color="red", linestyle="--", label=f"threshold={args.threshold}")
        ax.set_title(antigen)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.suptitle("Antigen specificity probability distributions", y=1.02)
    plt.tight_layout()
    prob_plot = os.path.join(args.output_dir, "multi_target_distributions.png")
    plt.savefig(prob_plot, bbox_inches="tight")
    plt.close()
    print(f"Saved distributions plot: {prob_plot}")

    # Plot 2: heatmap — sequences × antigens (capped at 200 rows for readability)
    prob_cols = [f"{a}_prob" for a in targets]
    sample = results[prob_cols].head(200) if len(results) > 200 else results[prob_cols]
    sample = sample.copy()
    sample.columns = targets
    plt.figure(figsize=(max(4, len(targets) * 1.5), min(12, len(sample) * 0.08 + 2)))
    sns.heatmap(sample.T, cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={"label": "Specificity probability"},
                xticklabels=False)
    plt.xlabel(f"Sequences (first {len(sample)} of {len(results)})")
    plt.ylabel("Antigen")
    plt.title("Per-sequence antigen specificity scores")
    plt.tight_layout()
    heatmap_path = os.path.join(args.output_dir, "multi_target_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
