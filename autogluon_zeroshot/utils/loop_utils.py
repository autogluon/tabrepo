# TODO: Code taken from https://github.com/Innixma/autogluon-benchmark/blob/master/autogluon_benchmark/benchmark_context/_output_suite_context.py
#  Consider avoiding code dupe and depend on `autogluon_benchmark` directly.

import ray


# TODO: Add docstrings
def with_seq(func: callable,
             input_list: list,
             input_is_dict: bool = True,
             kwargs=None,
             allow_exception=False,
             exception_default=None) -> list:
    """
    For-loop through a function call sequentially
    """
    if kwargs is None:
        kwargs = dict()
    if allow_exception:
        def _func(*args, **kw):
            try:
                return func(*args, **kw)
            except:
                return exception_default
    else:
        _func = func
    out_list = []
    for input_val in input_list:
        if input_is_dict:
            out_list.append(_func(**input_val, **kwargs))
        else:
            out_list.append(_func(input_val, **kwargs))
    return out_list


# TODO: Add docstrings
def with_ray(func: callable,
             input_list: list,
             input_is_dict: bool = True,
             kwargs=None,
             allow_exception=False,
             exception_default=None) -> list:
    """
    For-loop through a function call with ray
    """
    if kwargs is None:
        kwargs = dict()
    if allow_exception:
        def _func(*args, **kw):
            try:
                return func(*args, **kw)
            except:
                return exception_default
    else:
        _func = func

    if not ray.is_initialized():
        ray.init()
    remote_func = ray.remote(_func)
    results = []
    for i in input_list:
        if input_is_dict:
            results.append(remote_func.remote(**i, **kwargs))
        else:
            results.append(remote_func.remote(i, **kwargs))
    result = ray.get(results)
    return result
