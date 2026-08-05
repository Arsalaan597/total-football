# Total Football — Winner Model v0.1

## Overview
The **Winner Model v0.1** is a machine learning pipeline designed to predict English Premier League (EPL) match outcomes (`H` = Home Win, `D` = Draw, `A` = Away Win) using information known prior to kickoff.

To prevent **data leakage**, all feature engineering is calculated strictly using historical match data completed before the match date, using chronological rolling windows[cite: 1].

---

## Dataset & Leakage-Safe Feature Engineering

### 1. Raw Data Source
- **Source:** Historical Premier League match CSVs from [Football-Data.co.uk](https://www.football-data.co.uk/)
- **Seasons Included:** 2020/2021 through 2024/2025
- **Location:** `data/raw/` (`EPL_2020_2021.csv` to `EPL_2024_2025.csv`)

### 2. Core Features (Rolling Last 5 Matches)
Features are computed per team using a 1-match shift to ensure no current match stats leak into predictions[cite: 1]:
- `home_points_last_5` / `away_points_last_5`: Recent overall form points[cite: 1].
- `home_goals_for_last_5` / `away_goals_for_last_5`: Recent offensive output[cite: 1].
- `home_goals_against_last_5` / `away_goals_against_last_5`: Recent defensive record[cite: 1].
- `home_home_points_last_5` / `away_away_points_last_5`: Venue-specific form (Home team at home, Away team away)[cite: 1].

*Rows without 5 prior historical matches (early season fixtures) are dropped to maintain full feature integrity[cite: 1].*

---

## Model Architecture & Evaluation Strategy

- **Method:** Multinomial Logistic Regression pipeline using `scikit-learn` with `StandardScaler`[cite: 1].
- **Evaluation Split:** Chronological time-based split (Matches before `2023-08-01` for training; matches on/after `2023-08-01` for testing)[cite: 1].
- **Training Samples:** 990 matches
- **Testing Samples:** 740 matches

---

## Performance Results & Metrics

The model significantly outperforms the baseline (always predicting the most frequent training set class, `H`)[cite: 1].

| Model | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| **Baseline (Always Predict Home Win 'H')** | 43.78% | 0.2030 |
| **Winner Model v0.1 (Logistic Regression)** | **51.89%** | **0.3810** |

### Classification Report

```text
              precision    recall  f1-score   support

    Away (A)       0.52      0.51      0.52       249
    Draw (D)       0.00      0.00      0.00       167
    Home (H)       0.52      0.79      0.63       324

    accuracy                           0.52       740
   macro avg       0.35      0.43      0.38       740
weighted avg       0.40      0.52      0.45       740