# Total Football

A modular football analytics project built to learn machine learning from first principles.

## Version 1.0 goal

Given information available before a football match, build separate models for:

- Match result (home win, draw, or away win)
- Total goals
- Both teams to score (BTTS)
- Over/under 2.5 goals

A decision engine will later combine those predictions into one clear recommendation. Each component will be developed, evaluated, and documented separately before integration.

## Guiding principle

> Understand first. Optimize later.

We use a method only when we understand the problem it solves and why it fits this project.

## Project map

```text
data/       Original and prepared datasets (not committed)
docs/       Decisions, learning notes, and project documentation
notebooks/  Exploratory work and experiments
src/        Reusable application code
models/     Saved trained models (not committed)
reports/    Generated findings and visual outputs (not committed)
tests/      Automated checks for reusable code
```

## Current stage

Project foundation. The next lesson is to define the dataset: one row represents one completed match, and each column represents information that was known at the appropriate time.

## Getting started

This project will use Python 3.11 or newer. Dependency and environment instructions will be added when we introduce the first library, rather than installing tools before we know why we need them.

## Documentation

- [Project charter](docs/project_charter.md)
- [Initial data plan](docs/data_plan.md)
- [Winner model v0.1 brief](docs/winner_model_v0_1_brief.md)
