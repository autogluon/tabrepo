# TabArena

TabArena is a living benchmarking system that makes benchmarking tabular machine learning models a reliable experience. TabArena implements best practices to ensure methods are represented at their peak potential, including cross-validated ensembles, strong hyperparameter search spaces contributed by the method authors, early stopping, model refitting, parallel bagging, memory usage estimation, and more.

TabArena currently consists of:

- 51 manually curated tabular datasets representing real-world tabular data tasks.
- 9 to 30 evaluated splits per dataset.
- 16 tabular machine learning methods, including 3 tabular foundation models.
- 25,000,000 trained models across the benchmark, with all validation and test predictions cached to enable tuning and post-hoc ensembling analysis.
- A [live TabArena leaderboard](https://huggingface.co/spaces/TabArena/leaderboard) showcasing the results.

## Installation

To install TabArena, ensure you are using Python 3.9-3.11. Then, run the following:

```
git clone https://github.com/autogluon/tabrepo.git
pip install -e tabrepo/[benchmark]
```

## Quickstart

Please refer to our [example scripts](https://github.com/TabArena/tabarena_benchmarking_examples/tree/main) for using TabArena.

To locally reproduce individual configurations and compare with the TabArena results of those configurations, refer to [examples/tabarena/run_quickstart_tabarena.py](examples/tabarena/run_quickstart_tabarena.py).

To locally reproduce all tables and figures in the paper using the raw results data, run [examples/tabarena/run_tabarena_eval.py](examples/tabarena/run_tabarena_eval.py)

## Documentation

TabArena code is currently being polished. Documentation for TabArena will be available soon.

## Paper

The TabArena paper is currently under review, and will be made publicly available soon.

## TabRepo

To see details about TabRepo, the portfolio simulation repository, refer to [tabrepo.md](tabrepo.md).
