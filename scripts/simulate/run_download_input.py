from tabrepo.contexts import get_context


if __name__ == '__main__':
    context_name = 'D244_F3_C1416_3'  # The context you want to download
    include_zs = True  # Set False to only download files necessary for SingleBest (skip predict proba files)

    context = get_context(context_name)
    context.download(include_zs=include_zs, use_s3=False)
