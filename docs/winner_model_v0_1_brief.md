# Total Football — Winner Model v0.1 Brief

**Status:** agreed planning reference; implementation begins next session.

## The deadline outcome

Within two days, deliver one reproducible, working model that predicts a league match result from information known before kickoff. It must train on historical data, evaluate on later unseen matches, and return probabilities for home win, draw, and away win.

This is a learning project. A modest but correctly evaluated model is more valuable than an over-complicated model with hidden data leakage.

## Model definition

| Decision | v0.1 choice |
| --- | --- |
| First model | Winner prediction |
| Task type | Three-class classification |
| Target labels | `H` = home win, `D` = draw, `A` = away win |
| Unit of data | One completed league match per row |
| Prediction point | Before official starting line-ups are announced |
| Initial competition | One league only, selected after inspecting available data |
| Training method | Multinomial logistic regression in a preprocessing pipeline |
| Evaluation split | Older matches for training; newer matches for testing |
| Main outputs | Predicted class and probabilities for H, D, and A |

## Feature plan

All features are calculated separately for the home and away teams. Each value must use only matches completed before the row's match date.

### Core features — required

These form the first complete model:

```text
home_points_last_5
away_points_last_5
home_goals_for_last_5
away_goals_for_last_5
home_goals_against_last_5
away_goals_against_last_5
home_home_points_last_5
away_away_points_last_5
```

The first six capture recent overall form. The last two capture a team's recent performance in the same venue context: home teams at home, and away teams away from home.

### Stretch features — add only after the core pipeline works

```text
home_rest_days
away_rest_days
home_season_points_per_game
away_season_points_per_game
```

These are useful and still leakage-safe, but they add implementation and validation work. Add them only if the core model trains, evaluates, and predicts successfully first.

### Explicitly out of scope for v0.1

- Player form, injuries, and line-ups
- Head-to-head records
- Betting odds
- In-match statistics such as possession, shots, or xG
- Team names as model inputs
- A constant `home_advantage = 1` column

The last item is intentionally excluded: a constant has no variation, so it cannot teach the model anything. Keeping home-team statistics in `home_*` columns and away-team statistics in `away_*` columns already preserves the match roles.

## Non-negotiable data rules

1. A match may contribute to features for later matches, never to features for itself.
2. `FTHG`, `FTAG`, and final result are used to create targets and future historical form; they are never features for their own match.
3. Every rolling calculation is ordered by date and shifted so the current match is excluded.
4. The test period stays in the future relative to the training period. No random train/test split.
5. A test-period match may use information from earlier completed matches in the test period. That matches real life and is not leakage.
6. Rows without enough prior match history for a required feature are dropped in v0.1 rather than filled with invented values.
7. Keep raw data unchanged. Save derived datasets separately and document the source and seasons.

## Evaluation

We will report:

- Accuracy: proportion of correctly predicted results.
- Macro F1: gives home wins, draws, and away wins equal importance, even if draws are less common.
- Confusion matrix: shows which result types the model confuses.
- A simple baseline: always predict the most common training-set result.

The model must outperform—or at minimum be honestly compared against—the baseline. Accuracy alone is not enough.

## Definition of done

The v0.1 winner model is complete when all of the following are true:

1. Data source, league, and seasons are recorded.
2. A script or notebook creates the leakage-safe feature table.
3. The model trains successfully on the older period.
4. Evaluation metrics and a confusion matrix are saved or displayed.
5. At least one held-out match has its predicted class and three probabilities shown.
6. The README documents features, method, result, limitations, and how to rerun it.
7. Work is committed to Git in understandable steps.

## Two-day delivery plan

### Day 1 — data and features

1. Choose and inspect a historical league dataset.
2. Understand the raw columns and target labels.
3. Build and validate chronological, rolling features.
4. Save the model-ready table.

### Day 2 — model and evidence

1. Train the baseline and logistic-regression model.
2. Evaluate using the future test period.
3. Inspect mistakes and probabilities.
4. Document the outcome and commit the finished version.

## How we will work together

Use a short mode request at the beginning of any task or message.

| Mode | What I do | What you do |
| --- | --- | --- |
| `Teacher` | Explain a concept, ask you to reason, and review your answer before coding. | Learn and write the implementation. |
| `Assistant` | Help you plan, unblock errors, review code, and suggest the next concrete action. | Drive the keyboard and decisions. |
| `Pair` | Explain while we build one small piece at a time; I provide a skeleton only after you attempt it. | Type, run, and adapt the code. |
| `Direct help` | Give a focused answer or a complete snippet when time is critical. | Ask for this explicitly. |

Default mode: **Assistant**. You can set intensity too, for example: `Teacher — high guidance`, `Assistant — hints only`, or `Pair — moderate guidance`.

## Next-session starting point

Start with: `Assistant — moderate guidance. Begin Day 1: choose our dataset.`

Then we will select the source together, inspect the raw columns, and you will make the first implementation decisions.
