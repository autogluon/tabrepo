# Reproducing TabRepo

To reproduce TabRepo, we must:

1. Install AutoMLBenchmark
2. (TODO) Setup AWS credentials
3. Edit the `custom_configs/config.yaml` file
4. Execute the `run_zeroshot.sh` script
5. Aggregate the results
6. Re-run failed tasks
7. Generate the TabRepo artifacts from the results
8. Add the artifacts as a TabRepo Context
9. Run TabRepo with the new Context

## 1) Installation

On a fresh python virtual environment using Python 3.9 to 3.11:

```bash
# Create a fresh venv
python -m venv venv
source venv/bin/activate

# Clone AutoMLBenchmark with TabRepo configs specified
# Make sure to do this when in the directory above the `tabrepo` project, `tabrepo` and `automlbenchmark` should exist side-by-side.
git clone https://github.com/Innixma/automlbenchmark.git --branch 2023_12_07

# Install AutoMLBenchmark (https://openml.github.io/automlbenchmark/docs/getting_started/)
cd automlbenchmark
pip install --upgrade pip
pip install -r requirements.txt
```

You are all set!

## 2) Setup AWS Credentials

Due to the large amount of compute required to reproduce TabRepo, we will be using AWS.
We will need to ensure that `boto3` recognizes your AWS credentials. We need this in order to spin up the AWS EC2 instances via AutoMLBenchmark's "aws" mode.
You will need to ensure that `boto3` is working before moving to the next step.

## 3) Edit the `custom_configs/config.yaml` file

The file can be found here: `automlbenchmark/custom_configs/config.yaml`.

We need to edit this file so that it points to the correct bucket.

```yaml
  s3:                               # sub-namespace for AWS S3 service.
    bucket: automl-benchmark-ag        # must be unique im whole Amazon s3, max 40 chars, and include only numbers, lowercase characters and hyphens.
    root_key: ec2/2023_12_07/                  #
```

Replace the `bucket` argument above in the `config.yaml` file with a bucket you have created in your AWS account.
Note that this bucket must start with `automl-benchmark` in the name, otherwise it won't work.
This is the location all output artifacts will be saved to. You can also optionally change the root_key, which is the directory within the bucket.
Note that bucket names are globally unique, so you will need to create a new one.

## 4) Execute the `run_zeroshot.sh` script

Note: Currently the below script will run for upwards of 10 days before all results are complete, using a significant amount of compute.

Theoretically, it could use 1,370,304 hours of on-demand m6i.2xlarge compute (~$430,000, not including storage costs) if all machines ran for the full time limit (in practice it is over an order of magnitude lower).

```bash
# In the root directory, where `tabrepo` and `automlbenchmark` exist
mkdir execute_tabrepo
cd execute_tabrepo
../tabrepo/scripts/execute_benchmarks/run_zeroshot.sh
```

## 5) Aggregating Results

Once the benchmark has fully finished, you can aggregate the results by running this script: `https://github.com/Innixma/autogluon-benchmark/blob/master/scripts/aggregate_all.py`.
You will need to specify a version name matching the folder your results are saved to in S3, and specify `aggregate_zeroshot=True`.

## 6) Rerunning Failed Tasks

If you had transient failures and need to rerun failed tasks, we recommend contacting us for guidance, as this becomes non-trivial.

## 7) Generate TabRepo Artifacts

The TabRepo artifacts will automatically be generated when aggregating the results.

## 8) Adding artifacts as a new TabRepo Context

Please contact us to assist with adding your artifacts as a new TabRepo Context.

## 9) Run TabRepo on the new Context

Now that your context has been added to TabRepo, simply specify the context name to load it as a EvaluationRepository:

```python
from tabrepo import load_repository, EvaluationRepository
repo: EvaluationRepository = load_repository(context_name, cache=True)
repo.print_info()
```

You can also run the full suite of experiments with the following command:

```
!python tabrepo/scripts/baseline_comparison/evaluate_baselines.py --repo "{YOUR_CONTEXT_NAME}"
```
