from tabflow.cli.launch_jobs import JobManager
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata


"""
bash ./tabrepo/tabflow/docker/build_docker.sh tabarena tabarena-neerick 763104351884 097403188315 us-west-2
 
aws s3 cp --recursive "s3://prateek-ag/neerick-exp-3/" ../data/neerick-exp-3/ --exclude "*.log"


https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs
"""


docker_image_uri = "097403188315.dkr.ecr.us-west-2.amazonaws.com/tabarena:tabarena-neerick"
# docker_image_uri = "097403188315.dkr.ecr.us-west-2.amazonaws.com/pmdesai:mlflow-tabrepo"
sagemaker_role = "arn:aws:iam::097403188315:role/service-role/AmazonSageMaker-ExecutionRole-20250128T153145"


if __name__ == "__main__":
    task_metadata = load_task_metadata(subset="TabPFNv2")

    experiment_name = "neerick-exp-mitra_toy"
    max_concurrent_jobs = 100
    batch_size = 8
    wait = True
    s3_bucket = "prateek-ag"
    region_name = "us-west-2"
    instance_type = "ml.m6i.2xlarge"  # TODO: Need to use a GPU instance

    methods_file = "./tabrepo/tabflow/configs/configs_mitra.yaml"  # TODO: Need to create this file
    methods = JobManager.load_methods_from_yaml(methods_file=methods_file)

    datasets = list(task_metadata["name"])

    # toy run
    datasets = datasets[:1]
    folds = [0]
    repeats = [0]

    print(datasets)
    print()

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

    tasks_dense = job_manager.get_tasks_dense(
        datasets=datasets,
        repeats=repeats,
        folds=folds,
        methods=methods,
    )
    # uncached_tasks = job_manager.filter_to_only_uncached_tasks(tasks=tasks_dense, verbose=True)

    tasks_batch = job_manager.batch_tasks(tasks=tasks_dense, batch_size=batch_size)
    tasks_batch_combined = tasks_batch

    uncached_tasks_batched = tasks_batch_combined

    job_manager.run_tasks_batched(task_batch_lst=uncached_tasks_batched, check_cache=False)
