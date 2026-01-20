# Project DataPub Fraud Data Centric Preprocessing

## Project Purpose

This notebook applies a data-centric approach: instead of treating the dataset as fixed and only tuning models, it focuses on systematically transforming and cleaning the dataset so a supervised classifier can learn from it more reliably.

It also borrows the “centering” idea from \href{https://dcai.csail.mit.edu/2024/data-centric-model-centric/}{customer-centricity} (put the customer at the core of decisions) and applies it to data: decisions are driven by what makes the dataset easiest and safest for ML to consume.

This work was developed as part of a contest for **The Data Pub** initiative (Mexico-based).

`X_test_datapub.csv` is excluded during target-driven analysis because without the target you cannot validate checks like correlation against `fraud_flag`, so its use is not helpful at that stage.

---

## Repository Structure

```bash
DataPubDataCentricChallenge_AaronMartinCastilloMedina/
├── train/
│   ├── X_train_datapub.csv
│   └── Y_train_datapub.csv
├── test/
│   └── X_test_datapub.csv
├── results/
│   └── DataPubDataCentricChallenge_AaronMartinCastilloMedina.csv
├── DataPubDataCentricChallenge_AaronMartinCastilloMedina.ipynb
└── requirements.txt
```

---

## Folder and File Descriptions

### `train/`
Stores the labeled training data used for transformation, analysis, and model training.

- `X_train_datapub.csv`: Training features (input variables).
- `Y_train_datapub.csv`: Training labels (target), includes `fraud_flag`.

### `test/`
Stores the unlabeled test data used only to generate the final contest submission.

- `X_test_datapub.csv`: Test features (same schema as `X_train_datapub.csv`, without labels).

### `results/`
Stores the generated outputs for the challenge.

- `DataPubDataCentricChallenge_AaronMartinCastilloMedina.csv`: Submission file with predictions for the test set.

### `DataPubDataCentricChallenge_AaronMartinCastilloMedina.ipynb`
Main notebook containing the full workflow: loading data, feature transformation, cleaning, and generation of the submission CSV.

### `requirements.txt`
Lists the Python dependencies required to run the notebook.

---

## Goal and Procedure

Goal: build a training-ready table with numeric features that represent what matters for fraud prediction, then remove noise and redundancy.

Procedure:

- Start from the original training table: 92,790 rows × 146 columns.
- Replace the fixed “basket slots” structure (`item1..item24`) with one column per product plus `total_purchase`, so product info becomes model-friendly counts instead of strings. This creates a wide product-feature table (92,790 rows × 319 columns).
- Handle near-duplicate product columns using a cosine-similarity proposal: compare product-name strings and, if similarity > 0.60, merge the two product-count columns into one (sum counts) and drop the duplicate.
  - The similarity is computed with TF-IDF + cosine similarity (`cosine_sim`).
- Join features with the target and prune features:
  - Drop redundant predictors with absolute correlation = 1 (4 columns).
  - Drop weak predictors with low absolute correlation to `fraud_flag` using threshold 0.01.

---

## Results

Class balance is heavily imbalanced:

- `fraud_flag = 0`: 91,471
- `fraud_flag = 1`: 1,319

Final dataset shown after dropping steps: 92,790 rows × 19 columns.

Example of the strongest simple correlations with the target (still small): `item_computers`, `total_purchase`, `item_fulfilment_charge`.

---

## Takeaways

The cosine-similarity step is not just “mentioned,” it is used as a concrete merge rule: similar product-name columns are detected and fused (counts summed) using a >0.60 threshold.

Dimensional change shows the impact of data-centric cleaning:

- Original: 146 columns
- Final: 19 columns

Fraud is rare in the data, so later evaluation and training need imbalance-aware choices (metrics, class weights, sampling).

Simple correlations with `fraud_flag` are weak overall, so fraud likely depends on interactions or non-linear patterns, not one single feature.
