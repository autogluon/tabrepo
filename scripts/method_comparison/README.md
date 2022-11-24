# Method comparison

The setup is the same as for other experiments, you will just need in addition to install syne-tune to be able to run
searchers:

```
pip install syne-tune[extra]
```

Then you can run:
* `run_method_comparison.py`: to generate evaluations of different search strategies (all, random, local, zeroshot, zeroshot-ensemble)
* `plot_results_comparison.py`: to get a table with the results

Note that this assumes that you have obtained large files by running `run_download_zeroshot_pred_proba.py`, also the 
first time you call run `run_method_comparison.py` a conversion of `zeroshot_pred_proba_2022_10_13_zs.pkl` will be 
triggered to generate `zeroshot_pred_per_task` which is a format allowing for lazy evaluation (loading task data
on the fly).