
<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <img src="https://avatars.githubusercontent.com/u/210855230" width="175" alt="TabArena Logo"/>
    </summary>
  </ul>
</div>

## A  Living Benchmark for Machine Learning on Tabular Data 💫
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

### More Documentation
TabArena code is currently being polished. Documentation for TabArena will be available soon.

# 🪄 Installation

To install TabArena, ensure you are using Python 3.9-3.11. Then, run the following:

```
git clone https://github.com/autogluon/tabrepo.git
pip install -e tabrepo/[benchmark]
```

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

TabArena was built upon [TabRepo](https://arxiv.org/pdf/2311.02971) and now replaces TabRepo. To see details about TabRepo, the portfolio simulation repository, refer to [tabrepo.md](tabrepo.md).
