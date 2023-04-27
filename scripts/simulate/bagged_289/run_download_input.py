from autogluon_zeroshot.contexts import get_context


if __name__ == '__main__':
    context_name = 'BAG_D244_F10_C608_FULL'  # The context you want to download
    dry_run = True  # Set False to download files
    include_zs = True  # Set False to only download files necessary for SingleBest (skip predict proba files)

    if dry_run:
        print(f'NOTE: Files will not be downloaded as `dry_run=True`.\n'
              f'This will log what files will be downloaded instead.\n'
              f'Set `dry_run=False` to download the files.')

    context = get_context(context_name)
    context.download(include_zs=include_zs, dry_run=dry_run)
