
<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <img src="https://avatars.githubusercontent.com/u/210855230" width="175" alt="TabArena Logo"/>
    </summary>
  </ul>
</div>

## A Living Benchmark for Machine Learning on Tabular Data 💫

---

| 🚀 [Leaderboard](https://huggingface.co/spaces/TabArena/leaderboard) | 📂 [Example Scripts](https://github.com/TabArena/tabarena_benchmarking_examples/tree/main) | 📊 [Dataset Curation](https://github.com/TabArena/tabarena_dataset_curation) | 📄 [ArXiv Paper](https://arxiv.org/abs/2506.16791) |
|:-------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|

---
</div>

TabArena is a living benchmarking system that makes benchmarking tabular machine learning models a reliable experience. TabArena implements best practices to ensure methods are represented at their peak potential, including cross-validated ensembles, strong hyperparameter search spaces contributed by the method authors, early stopping, model refitting, parallel bagging, memory usage estimation, and more.

TabArena currently consists of:

- 51 manually curated tabular datasets representing real-world tabular data tasks.
- 9 to 30 evaluated splits per dataset.
- 16 tabular machine learning methods, including 3 tabular foundation models.
- 25,000,000 trained models across the benchmark, with all validation and test predictions cached to enable tuning and post-hoc ensembling analysis.
- A [live TabArena leaderboard](https://huggingface.co/spaces/TabArena/leaderboard) showcasing the results.


## 🕹️ Quickstart

### Benchmarking and Running TabArena Models
Please refer to our [example scripts](https://github.com/TabArena/tabarena_benchmarking_examples/tree/main) for using TabArena.

### Datasets 
Please refer to our [dataset curation repository](https://github.com/TabArena/tabarena_dataset_curation) to learn more about or contributed data! 

### Evaluation & Reproducing Results
To locally reproduce individual configurations and compare with the TabArena results of those configurations, refer to [examples/tabarena/run_quickstart_tabarena.py](examples/tabarena/run_quickstart_tabarena.py).

To locally reproduce all tables and figures in the paper using the raw results data, run [examples/tabarena/run_generate_paper_figures.py](examples/tabarena/run_generate_paper_figures.py)

To locally generate the latest results leaderboard, run [examples/tabarena/run_generate_main_leaderboard.py](examples/tabarena/run_generate_main_leaderboard.py)

### More Documentation
TabArena code is currently being polished. Detailed Documentation for TabArena will be available soon.

# 🪄 Installation

To install TabArena, ensure you are using Python 3.9-3.12. Then, run the following:

## Evaluation (Leaderboard / Metrics)

If you don't intend to fit models, this is the simplest installation.

```
git clone https://github.com/autogluon/tabarena.git
pip install -e tabarena/
```

## Benchmark (Fitting Models)

If you intend to fit models, this is required.

```
git clone https://github.com/autogluon/tabarena.git
pip install -e tabarena/[benchmark]

# use GIT_LFS_SKIP_SMUDGE=1 in front of the command if installing TabDPT fails due to a broken LFS/pip setup
# GIT_LFS_SKIP_SMUDGE=1 uv pip install -e tabarena/[benchmark]
```

## Developer Install

With this installation, you will have the latest version of AutoGluon in editable form.

```
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh

git clone https://github.com/autogluon/tabarena.git
pip install -e tabarena/[benchmark]
```

## Example Install + Run

```
pip install uv
uv init -p 3.11
uv sync
git clone https://github.com/autogluon/tabarena.git
uv pip install -e tabarena/[benchmark]
git clone git@github.com:TabArena/tabarena_benchmarking_examples.git
cd tabarena_benchmarking_examples/tabarena_minimal_example/custom_tabarena_model 
python run_tabarena_lite.py 
```

# Downloading and using TabArena Artifacts

Artifacts will by default be downloaded into `~/.cache/tabarena/`. You can change this by specifying the environment variable `TABARENA_CACHE`.

The types of artifacts are:

1. Raw data -> The original results that are used to derive all other artifacts. Contains per-child test predictions from the bagged models, along with detailed metadata and system information absent from the processed results. Very large, often 100 GB per method type.
2. Processed data -> The minimal information needed for simulating HPO, portfolios, and generating the leaderboard. Often 10 GB per method type.
3. Results -> Pandas DataFrames of the results for each config and HPO setting on each task. Contains information such as test error, validation error, train time, and inference time. Generated from processed data. Used to generate leaderboards. Very small, often under 1 MB per method type.
4. Leaderboards -> Aggregated metrics comparing methods. Contains information such as ELO, win-rate, average rank, and improvability. Generated from a list of results files. Under 1 MB for all methods.
5. Figures & Plots -> Generated from results and leaderboards.

## Raw Data

Refer to [examples/tabarena/inspect_raw_data.py](examples/tabarena/inspect_raw_data.py)

## Processed Data

Refer to [examples/tabarena/inspect_processed_data.py](examples/tabarena/inspect_processed_data.py)

## Results

Instructions TBD

# 📄 Publication for TabArena

If you use TabArena in a scientific publication, we would appreciate a reference to the following paper:

**TabArena: A Living Benchmark for Machine Learning on Tabular Data**, 
Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, Frank Hutter, Preprint., 2025

Link to publication: [arXiv](https://arxiv.org/abs/2506.16791)

Bibtex entry:
```bibtex
@article{erickson2025tabarena,
  title={TabArena: A Living Benchmark for Machine Learning on Tabular Data}, 
  author={Nick Erickson and Lennart Purucker and Andrej Tschalzev and David Holzmüller and Prateek Mutalik Desai and David Salinas and Frank Hutter},
  year={2025},
  journal={arXiv preprint arXiv:2506.16791},
  url={https://arxiv.org/abs/2506.16791}, 
}
```


--- 
## Relation to TabRepo 

TabArena was built upon and now replaces [TabRepo](https://arxiv.org/pdf/2311.02971). To see details about TabRepo, the portfolio simulation repository, refer to [tabrepo.md](tabrepo.md).
