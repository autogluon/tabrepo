# Example for Running TabArena on One (Private/Local) Dataset with TabFlow-SLURM

This code contains an example of using the TabArena library to benchmark SOTA ML method
on one private/local dataset for biopsy predictions task. The local dataset is not
shared online. Moreover, we focus here on running the benchmarking of this dataset
across a SLURM cluster. 

## Overview

* `get_local_task.py` - Example of how to read a private/local dataset such that it can
  be used with TabArena
* `run_for_one_dataset_on_slurm.py` - Example for running with TabFlow-SLURM.
* `run_eval_from_slurm_results.py` - Evaluation code for results from TabFlow-SLURM.

**WARNING: the plotting and evaluation code of TabArena is not streamlined for evaluating
just one dataset yet. We plan to support this with a less code in the future but for
now, the plotting code is quite complicated.**

## Install

Follow the setup of of TabFlow-SLURM's README file and make sure to configure
the `setup_slurm_base.py` as well.

## Usage

One can schedule the job on a SLURM cluster with:

```bash
# active your venv and cd to the directory of the script
source /work/dlclarge2/purucker-tabarena/venvs/tabarena_ag14/bin/activate && cd /work/dlclarge2/purucker-tabarena/code/tabarena_benchmarking_examples/tabarena_applications/biopsy_predictions
# now follow the output of run_on_slurm.py
```

