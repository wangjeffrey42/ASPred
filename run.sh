#!/bin/bash
cd "$(dirname "$0")"

if [[ "$CONDA_DEFAULT_ENV" != "aspred" ]]; then
    echo "Run 'conda activate aspred' first."
    exit 1
fi

clear
echo "ASPred — Antibody Specificity Predictor"
echo "─────────────────────────────────────────"
echo "What do you want to do?"
echo ""
echo "  1) Test antibodies against ONE virus"
echo "  2) Test antibodies against ALL viruses (FLU + HIV + COVID)"
echo "  3) Check how accurate the predictions were"
echo "  4) Quit"
echo ""
read -p "> " choice

# ── option 1 ───────────────────────────────────────────────
if [[ "$choice" == "1" ]]; then

    echo ""
    echo "Which virus do you want to test binding against?"
    echo "(f1 = more accurate, loss = alternative version)"
    echo ""
    MDL_PATHS=(); i=0
    for dir in models/*/training_output; do
        [[ -f "$dir/adapter_model.safetensors" ]] || continue
        name=$(basename "$(dirname "$dir")")
        # Make a readable label
        label=$(echo "$name" | sed 's|650M_heavy_||;s|_f1| (f1)|;s|_loss| (loss)|')
        echo "  $((i+1))) $label"
        MDL_PATHS+=("$dir"); ((i++))
    done
    read -p "> " n
    SELECTED_MODEL="${MDL_PATHS[$((n-1))]}"
    [[ -z "$SELECTED_MODEL" ]] && echo "Invalid." && exit 1

    echo ""
    echo "Which antibody sequences do you want to test?"
    echo "(these are the amino acid sequences the model will score)"
    echo ""
    DS_PATHS=(); DS_NAMES=(); i=0
    for f in datasets/*/test.csv; do
        name=$(echo "$f" | sed 's|datasets/||;s|/test.csv||')
        echo "  $((i+1))) $name"
        DS_PATHS+=("$f"); DS_NAMES+=("$name"); ((i++))
    done
    read -p "> " n
    INPUT_CSV="${DS_PATHS[$((n-1))]}"
    [[ -z "$INPUT_CSV" ]] && echo "Invalid." && exit 1

    read -p "Confidence threshold — sequences above this are called binders (default 0.5): " thr
    thr=${thr:-0.5}
    read -p "Output folder (default results/): " out; out=${out:-results/}

    # Build a readable label from the model name e.g. FLU_f1
    model_label=$(basename "$(dirname "$SELECTED_MODEL")" | sed 's|650M_heavy_||')

    echo ""
    python3 src/inference.py \
        --model_path    "$SELECTED_MODEL" \
        --input_csv     "$INPUT_CSV" \
        --threshold     "$thr" \
        --output_dir    "$out" \
        --output_prefix "${model_label}"

# ── option 2 ───────────────────────────────────────────────
elif [[ "$choice" == "2" ]]; then

    echo ""
    echo "Which antibody sequences do you want to test?"
    echo "(each sequence will be scored against FLU, HIV, and COVID)"
    echo ""
    DS_PATHS=(); i=0
    for f in datasets/*/test.csv; do
        name=$(echo "$f" | sed 's|datasets/||;s|/test.csv||')
        echo "  $((i+1))) $name"
        DS_PATHS+=("$f"); ((i++))
    done
    read -p "> " n
    INPUT_CSV="${DS_PATHS[$((n-1))]}"
    [[ -z "$INPUT_CSV" ]] && echo "Invalid." && exit 1

    read -p "Confidence threshold (default 0.5): " thr; thr=${thr:-0.5}
    read -p "Output folder (default results/multi_target/): " out; out=${out:-results/multi_target/}
    echo ""
    python3 src/multi_target_inference.py \
        --input_csv "$INPUT_CSV" \
        --base_dir  . \
        --threshold "$thr" \
        --output_dir "$out"

# ── option 3 ───────────────────────────────────────────────
elif [[ "$choice" == "3" ]]; then

    echo ""
    echo "Pick the dataset that has the real known answers:"
    echo "(this is what the model's predictions will be compared against)"
    echo ""
    DS_PATHS=(); DS_NAMES=(); i=0
    for f in datasets/*/test.csv; do
        name=$(echo "$f" | sed 's|datasets/||;s|/test.csv||')
        echo "  $((i+1))) $name"
        DS_PATHS+=("$f"); DS_NAMES+=("$name"); ((i++))
    done
    read -p "> " n
    INPUT_CSV="${DS_PATHS[$((n-1))]}"
    DATASET_NAME="${DS_NAMES[$((n-1))]}"
    [[ -z "$INPUT_CSV" ]] && echo "Invalid." && exit 1

    echo ""
    echo "Pick the predictions file to check against $DATASET_NAME:"
    echo ""
    PRED_PATHS=(); j=0

    # Pre-computed results from model training (models/*/test_evaluation/)
    for f in models/*/test_evaluation/test_predictions.csv; do
        [[ -f "$f" ]] || continue
        label=$(echo "$f" | sed 's|models/||;s|/test_evaluation/test_predictions.csv||;s|650M_heavy_||')
        echo "  $((j+1))) $label"
        PRED_PATHS+=("$f"); ((j++))
    done

    # User-generated results from running inference
    while IFS= read -r -d '' f; do
        [[ "$f" == *"multi_target"* ]] && continue
        label=$(echo "$f" | sed 's|results/||;s|__thresh[^/]*_predictions\.csv||;s|/test||;s|_predictions||')
        [[ -z "$label" ]] && label=$(basename "$f" | sed 's|__thresh.*||;s|_predictions||')
        echo "  $((j+1))) $label"
        PRED_PATHS+=("$f"); ((j++))
    done < <(find results/ -name "*_predictions.csv" -print0 2>/dev/null | sort -z)

    [[ ${#PRED_PATHS[@]} -eq 0 ]] && echo "No predictions found. Run option 1 first." && exit 1
    read -p "> " n
    PRED_CSV="${PRED_PATHS[$((n-1))]}"
    [[ -z "$PRED_CSV" ]] && echo "Invalid." && exit 1

    echo ""
    python3 - <<EOF
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import sys

truth = pd.read_csv("$INPUT_CSV")
preds = pd.read_csv("$PRED_CSV")

# Support both column naming conventions:
#   inference.py output:          predicted_label, prob_class1
#   pre-computed training output: pred, prob
if "predicted_label" in preds.columns and "prob_class1" in preds.columns:
    pred_col, prob_col = "predicted_label", "prob_class1"
elif "pred" in preds.columns and "prob" in preds.columns:
    pred_col, prob_col = "pred", "prob"
else:
    print("Error: this predictions file is not compatible with accuracy checking.")
    print("Make sure you ran single-model inference (option 1), not multi-target.")
    sys.exit(1)

if len(truth) != len(preds):
    print(f"Error: ground truth has {len(truth)} sequences but predictions file has {len(preds)}.")
    print("Make sure you picked the matching dataset and predictions file.")
    sys.exit(1)

y_true = truth["label"]
y_pred = preds[pred_col]
y_prob = preds[prob_col]

print(f"Antibodies tested:  {len(y_true)}")
print(f"Accuracy:           {accuracy_score(y_true, y_pred):.1%}  (overall correct predictions)")
print(f"F1 Score:           {f1_score(y_true, y_pred):.1%}  (balance of precision and recall)")
print(f"Precision:          {precision_score(y_true, y_pred):.1%}  (of predicted binders, how many truly bind)")
print(f"Recall:             {recall_score(y_true, y_pred):.1%}  (of true binders, how many were found)")
print(f"ROC-AUC:            {roc_auc_score(y_true, y_prob):.1%}  (overall discrimination ability)")
EOF

elif [[ "$choice" == "4" ]]; then
    exit 0
else
    echo "Invalid."
    exit 1
fi
