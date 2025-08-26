from tabflow.cli.launch_jobs import JobManager
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext

"""
# 1. Build the docker (ensure you use your own docker name to avoid overwriting other user's docker containers
bash ./tabrepo/tabflow/docker/build_docker.sh tabarena tabarena-neerick 763104351884 097403188315 us-west-2
"""
docker_image_uri = "097403188315.dkr.ecr.us-west-2.amazonaws.com/tabarena:tabarena-neerick"

"""
# 2. Need to set aws default region to us-west-2
aws configure
Default region name [None]: us-west-2
less ~/.aws/config
# delete config to revert
TODO: Make this not be required
"""

"""
# 3. Create a sagemaker role (only for fresh AWS account)
TODO
"""
sagemaker_role = "arn:aws:iam::097403188315:role/service-role/AmazonSageMaker-ExecutionRole-20250128T153145"

"""
# 4. Run this script (working directory should be one level above the root of `tabrepo`)

# 5. View the jobs and check their logs
https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs

# 6. View the result artifacts
https://us-west-2.console.aws.amazon.com/s3/buckets/{s3_bucket}?prefix={experiment_name}/
e.g: https://us-west-2.console.aws.amazon.com/s3/buckets/prateek-ag?prefix=tabarena-lightgbm-demo/

# 7. Download the artifacts to local
aws s3 cp --recursive "s3://prateek-ag/tabarena-lightgbm-demo" ../data/tabarena-lightgbm-demo/ --exclude "*.log"

# 8. Aggregate the local artifacts and evaluate them
TODO
"""


if __name__ == "__main__":
    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata.copy()  # metadata about the available datasets

    experiment_name = "tabarena-lightgbm-demo"  # The experiment name, used as the s3 path prefix for the saved files.
    s3_bucket = "prateek-ag"  # The s3 bucket to save results to

    max_concurrent_jobs = 10  # the max number of instances running jobs at the same time (values 1 - 15000)
    batch_size = 1  # The number of jobs to give to a single instance to run sequentially.
    wait = True  # If True, will keep the process alive until all jobs are finished launching.
    ignore_cache = False  # If True, will overwrite prior results in s3. If False, will skip runs for finished jobs.

    region_name = "us-west-2"  # AWS region to create instances. Keep as us-west-2.
    instance_type = "ml.m6i.2xlarge"  # options: ml.m6i.2xlarge (cpu, 15k cap), ml.g6.2xlarge (gpu, 400 cap)

    methods_file = "./tabrepo/tabflow/configs/configs_lightgbm_demo.yaml"
    methods = JobManager.load_methods_from_yaml(methods_file=methods_file)

    datasets = list(task_metadata["name"])
    toy_run = True

    # toy run
    if toy_run:
        # only run 1 dataset
        # only run 1 fold and 1 repeat per dataset
        datasets = datasets[:1]
        task_metadata["n_folds"] = 1
        task_metadata["n_repeats"] = 1

    job_manager = JobManager(
        experiment_name=experiment_name,
        task_metadata=task_metadata,
        methods_file=methods_file,
        max_concurrent_jobs=max_concurrent_jobs,
        s3_bucket=s3_bucket,
        wait=wait,
        instance_type=instance_type,
        batch_size=batch_size,
        sagemaker_role=sagemaker_role,
        docker_image_uri=docker_image_uri,
    )

    # get the list of jobs to execute
    tasks_batched = job_manager.get_tasks_batched(
        datasets=datasets,
        methods=methods,
        batch_size=batch_size,
        ignore_cache=ignore_cache,
    )

    # run the jobs
    job_manager.run_tasks_batched(task_batch_lst=tasks_batched)
