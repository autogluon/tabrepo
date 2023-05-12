# Baselines comparison

This directory contains script that allows to evaluate the following baselines:
* AutoGluon with best/high/medium presets
* Extensive search among all configurations (608 in the current example)
* Zeroshot portfolio of N configs


**Running evaluations.**

You can run:

```
python scripts/baseline_comparison/evaluate_baselines.py --expname {TAG}
```

which will evaluate all baselines and save intermediate results in `{ZEROSHOT_DIR}/data/results-baselines-comparison/{TAG}/`.

Once results are obtained, several results such as average rank and normalized score will be displayed
as well as the distribution of those two metrics across all tasks.

Results of baselines are cached in intermediate files and are not reevaluated by default if you rerun the script which
allows to add new method and avoid the need of reevaluating all baselines.
For instance,
```{ZEROSHOT_DIR}/data/results-baseline-comparison/{TAG}-folds_all-datasets_all/automl-baselines-v2-folds_all-datasets_all.csv.zip```
for automl-baselines defined in `baselines.py`.

To regenerate the result from scratch, you can either delete the file or set the flag `--ignore_cache`.