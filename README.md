# Project DataPub Fraud Data Centric Preprocessing

## Purpose

This notebook applies a data-centric approach: instead of treating the dataset as fixed and only tuning models, it focuses on systematically transforming and cleaning the dataset so a supervised classifier can learn from it more reliably.

It also borrows the “centering” idea from customer-centricity (put the customer at the core of decisions) and applies it to data: decisions are driven by what makes the dataset easiest and safest for ML to consume.

This work was developed as part of a contest for **The Data Pub** initiative (Mexico-based).

`X_test_datapub.csv` is excluded during target-driven analysis because without the target you cannot validate checks like correlation against `fraud_flag`, so its use is not helpful at that stage.

Reference: [1]

## Project Structure

These are the main files and folders used in the project. They define where the input data lives, where the output prediction file is saved, and where the notebook and dependencies are stored.

- `train/X_train_datapub.csv`  
  Training features (input variables). Used to build and validate the transformations and later train a model.

- `train/Y_train_datapub.csv`  
  Training labels (target). Contains `fraud_flag` for the rows in `X_train_datapub.csv`.

- `test/X_test_datapub.csv`  
  Test features (same format as `X_train_datapub.csv` but without labels). Used at the end to generate the final submission file.

- `results/DataPubDataCentricChallenge_AaronMartinCastilloMedina.csv`  
  Output predictions file (submission). This is the CSV generated for upload to the contest platform.

- `DataPubDataCentricChallenge_AaronMartinCastilloMedina.ipynb`  
  Main notebook. Contains the full workflow: loading data, transforming features, cleaning, and producing the final CSV.
  
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

## Results

Class balance is heavily imbalanced:

- `fraud_flag = 0`: 91,471
- `fraud_flag = 1`: 1,319

Final dataset shown after dropping steps: 92,790 rows × 19 columns.

Example of the strongest simple correlations with the target (still small): `item_computers`, `total_purchase`, `item_fulfilment_charge`.

## Takeaways

The cosine-similarity step is not just “mentioned,” it is used as a concrete merge rule: similar product-name columns are detected and fused (counts summed) using a >0.60 threshold.

Dimensional change shows the impact of data-centric cleaning:

- Original: 146 columns
- Final: 19 columns

Fraud is rare in the data, so later evaluation and training need imbalance-aware choices (metrics, class weights, sampling).

Simple correlations with `fraud_flag` are weak overall, so fraud likely depends on interactions or non-linear patterns, not one single feature.

- `requirements.txt`  
  Python dependencies needed to run the notebook (libraries and versions).

## References

[1] https://dcai.csail.mit.edu/2024/data-centric-model-centric/
