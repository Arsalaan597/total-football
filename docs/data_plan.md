# Initial Data Plan

## Unit of observation

For the first models, **one row is one completed match**. The row will contain the match outcome and a set of inputs that could have been known before kickoff.

## Targets we will derive

From a match's home and away goals, we can derive:

- **Result:** home win, draw, or away win
- **Total goals:** home goals + away goals
- **BTTS:** whether both teams scored at least once
- **Over 2.5:** whether total goals were greater than 2.5

We will derive these targets ourselves from the raw scores so the definitions are transparent and consistent.

## First data principles

1. Keep downloaded data unchanged in `data/raw/`.
2. Write any cleaned, model-ready dataset to `data/processed/`.
3. Record the source, league, seasons, and download date for every dataset.
4. Prevent data leakage: do not use information that only became known after kickoff when making a pre-match prediction.
5. Split data by time, not randomly, when evaluating predictive models. In reality, we always predict future matches from past matches.

## First decision still to make

Choose the league(s), seasons, and source for the initial dataset. We will evaluate sources before downloading anything.
