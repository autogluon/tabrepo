from autogluon.common.loaders import load_s3


if __name__ == '__main__':
    s3_bucket = "autogluon-zeroshot"
    s3_prefix = "tabpfn/tabforestpfn/"

    filenames = [
        "tabpfn.pt",
        "tabforest.pt",
        "tabforestpfn.pt",
    ]

    output_prefix = "/home/ubuntu/workspace/tabpfn_weights/"

    for filename in filenames:
        load_s3.download(s3_bucket, f"{s3_prefix}{filename}", f"{output_prefix}/{filename}")
