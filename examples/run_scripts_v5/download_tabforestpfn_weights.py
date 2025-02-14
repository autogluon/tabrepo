from autogluon.common.loaders import load_s3


if __name__ == '__main__':
    output_prefix = "/home/ubuntu/workspace/tabpfn_weights/"
    s3_bucket = "autogluon-zeroshot"
    s3_prefix = "tabpfn/tabforestpfn/"

    filenames = [
        "tabpfn.pt",
        "tabforest.pt",
        "tabforestpfn.pt",
    ]

    for filename in filenames:
        load_s3.download(s3_bucket, f"{s3_prefix}{filename}", f"{output_prefix}/{filename}")

    s3_bucket = "mmts-tb"
    s3_prefix = "tabular/TabPFN_mix_7_models/"

    filenames = [
        "TabPFN_mix_7_step_500000.pt",
        "TabPFN_mix_7_step_600000.pt",
        "TabPFN_mix_7_step_300000.pt",
    ]

    for filename in filenames:
        load_s3.download(s3_bucket, f"{s3_prefix}{filename}", f"{output_prefix}/{filename}")

    s3_bucket = "autogluon-zeroshot"
    s3_prefix = "tabpfn/tabdpt/"
    output_prefix = "/home/ubuntu/workspace/tabdpt_weights/"

    filenames = [
        "tabdpt_76M.ckpt",
    ]

    for filename in filenames:
        load_s3.download(s3_bucket, f"{s3_prefix}{filename}", f"{output_prefix}/{filename}")

    s3_bucket = "mmts-tb"
    s3_prefix = "tabular/"

    filenames = [
        "TabPFN_real_mix_7_models/model_step_500000.pt",
    ]

    for filename in filenames:
        load_s3.download(s3_bucket, f"{s3_prefix}{filename}", f"{output_prefix}/{filename}")
