import time
from typing import Tuple

import numpy as np

from autogluon.core.metrics import log_loss
from sklearn.metrics import log_loss as sk_log_loss
from autogluon_zeroshot.metrics._fast_log_loss import \
    fast_log_loss_end_to_end, fast_log_loss, extract_true_class_prob
from sklearn.preprocessing import normalize


def generate_y_true_and_y_pred_proba(num_samples, num_classes, random_seed=0):
    np.random.seed(seed=random_seed)
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = np.random.rand(num_samples, num_classes)
    y_pred = normalize(y_pred, axis=1, norm='l1')
    return y_true, y_pred


def get_eval_speed(*,
                   eval_metric: callable,
                   y_true: np.array,
                   y_pred: np.array,
                   num_repeats: int) -> Tuple[float, float]:
    ts = time.time()
    for _ in range(num_repeats):
        score = eval_metric(y_true, y_pred)
    te = time.time()
    time_average_s = (te - ts) / num_repeats
    return time_average_s, score


def benchmark_log_loss(num_samples: int, num_classes: int, num_repeats: int):
    """
    Benchmarks 4 log_loss computing methods, verifying equivalent scores and comparing compute speed

    1. sk_log_loss       : sklearn log_loss, which will be our baseline
    2. ag_log_loss       : AutoGluon's default log_loss implementation, which is a slightly faster variant to sklearn.
    3. fast_log_loss_e2e : The end-to-end version of fast log loss.
        Takes as input the same y_true and y_pred as the above metrics.
        This can be seen as the time taken to preprocess the data plus the time to pass to `fast_log_loss`.
    4. fast_log_loss     : The fully optimized fast log loss implementation.
        This is a very fast implementation whose run-time does not scale with num_classes.
        Ignores data transformation time, which is valid when re-using y_pred and y_true across many metric calls.
        y_pred and y_true are re-used many times in the greedy weighted ensemble fit, and thus this technique is valid.
        Technically for the purposes of ZS simulation, we can run the preprocessing logic on all y_pred for all models,
        thus never paying the cost of preprocessing and massively reducing memory usage.
    """
    print(f'Benchmarking log_loss... (num_samples={num_samples}, num_classes={num_classes}, num_repeats={num_repeats}')
    y_true, y_pred = generate_y_true_and_y_pred_proba(num_samples=num_samples, num_classes=num_classes)

    benchmark_metrics = [
        (sk_log_loss, 'sk_log_loss'),
        (log_loss.error, 'ag_log_loss'),
        (fast_log_loss_end_to_end.error, 'fast_log_loss_e2e')
    ]

    baseline_speed = None
    baseline_score = None

    for eval_metric, func_name in benchmark_metrics:
        time_average_s, score = get_eval_speed(
            eval_metric=eval_metric,
            y_true=y_true,
            y_pred=y_pred,
            num_repeats=num_repeats,
        )
        if baseline_speed is None:
            baseline_speed = time_average_s
        if baseline_score is None:
            baseline_score = score
        relative_speedup = baseline_speed / time_average_s
        print(f'\tTime = {time_average_s*1000:.4f} ms\t'
              f'| Score = {score}\t'
              f'| Rel Speedup = {relative_speedup:.1f}x\t| {func_name}')
        np.testing.assert_allclose(baseline_score, score)

    # fast log loss
    func_name = 'fast_log_loss'

    # The time this takes can be ignored, as for our purposes this is only paid once,
    # but y_true_opt and y_pred_opt are re-used in many log_loss calls.
    y_pred_opt = extract_true_class_prob(y_true=y_true, y_pred=y_pred)

    time_average_s, score = get_eval_speed(
        eval_metric=fast_log_loss.error,
        y_true=y_true,
        y_pred=y_pred_opt,
        num_repeats=num_repeats,
    )
    relative_speedup = baseline_speed / time_average_s
    print(f'\tTime = {time_average_s * 1000:.4f} ms\t'
          f'| Score = {score}\t'
          f'| Rel Speedup = {relative_speedup:.1f}x\t| {func_name}')
    np.testing.assert_allclose(baseline_score, score)


if __name__ == '__main__':
    """
    Run a benchmark demonstrating the result equivalence of `fast_log_loss` with normal `log_loss`,
    but computed much faster.
    
    Below is an example output using an m6i.32xlarge machine. fast_log_loss is at minimum 46x faster than sk_log_loss.
    
    Benchmarking log_loss... (num_samples=100, num_classes=2, num_repeats=1000
        Time = 0.2790 ms	| Score = 0.9403822961581705	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 0.2814 ms	| Score = 0.9403822961581705	| Rel Speedup = 1.0x	| ag_log_loss
        Time = 0.0142 ms	| Score = 0.9403822961581705	| Rel Speedup = 19.6x	| fast_log_loss_e2e
        Time = 0.0060 ms	| Score = 0.9403822961581705	| Rel Speedup = 46.5x	| fast_log_loss
    Benchmarking log_loss... (num_samples=1000, num_classes=2, num_repeats=1000
        Time = 0.3571 ms	| Score = 0.8685237331905123	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 0.3496 ms	| Score = 0.8685237331905123	| Rel Speedup = 1.0x	| ag_log_loss
        Time = 0.0631 ms	| Score = 0.8685237331905123	| Rel Speedup = 5.7x	| fast_log_loss_e2e
        Time = 0.0072 ms	| Score = 0.8685237331905123	| Rel Speedup = 49.9x	| fast_log_loss
    Benchmarking log_loss... (num_samples=1000, num_classes=10, num_repeats=1000
        Time = 0.4557 ms	| Score = 2.579619897673683	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 0.4308 ms	| Score = 2.579619897673683	| Rel Speedup = 1.1x	| ag_log_loss
        Time = 0.0625 ms	| Score = 2.579619897673683	| Rel Speedup = 7.3x	| fast_log_loss_e2e
        Time = 0.0073 ms	| Score = 2.579619897673683	| Rel Speedup = 62.5x	| fast_log_loss
    Benchmarking log_loss... (num_samples=1000, num_classes=100, num_repeats=1000
        Time = 1.2016 ms	| Score = 4.909232424969079	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 1.1516 ms	| Score = 4.909232424969079	| Rel Speedup = 1.0x	| ag_log_loss
        Time = 0.0625 ms	| Score = 4.909232424969079	| Rel Speedup = 19.2x	| fast_log_loss_e2e
        Time = 0.0072 ms	| Score = 4.909232424969079	| Rel Speedup = 166.5x	| fast_log_loss
    Benchmarking log_loss... (num_samples=10000, num_classes=2, num_repeats=100
        Time = 1.3963 ms	| Score = 0.8916323677552837	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 0.9198 ms	| Score = 0.8916323677552837	| Rel Speedup = 1.5x	| ag_log_loss
        Time = 0.6123 ms	| Score = 0.8916323677552837	| Rel Speedup = 2.3x	| fast_log_loss_e2e
        Time = 0.0176 ms	| Score = 0.8916323677552837	| Rel Speedup = 79.5x	| fast_log_loss
    Benchmarking log_loss... (num_samples=10000, num_classes=10, num_repeats=100
        Time = 2.5774 ms	| Score = 2.589909918127668	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 1.9414 ms	| Score = 2.589909918127668	| Rel Speedup = 1.3x	| ag_log_loss
        Time = 0.6366 ms	| Score = 2.589909918127668	| Rel Speedup = 4.0x	| fast_log_loss_e2e
        Time = 0.0166 ms	| Score = 2.589909918127668	| Rel Speedup = 155.4x	| fast_log_loss
    Benchmarking log_loss... (num_samples=10000, num_classes=100, num_repeats=100
        Time = 10.4843 ms	| Score = 4.898674510017278	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 8.8357 ms	| Score = 4.898674510017278	| Rel Speedup = 1.2x	| ag_log_loss
        Time = 0.6318 ms	| Score = 4.898674510017278	| Rel Speedup = 16.6x	| fast_log_loss_e2e
        Time = 0.0174 ms	| Score = 4.898674510017278	| Rel Speedup = 602.6x	| fast_log_loss
    Benchmarking log_loss... (num_samples=100000, num_classes=2, num_repeats=10
        Time = 12.2029 ms	| Score = 0.8839096911507552	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 6.6070 ms	| Score = 0.8839096911507552	| Rel Speedup = 1.8x	| ag_log_loss
        Time = 6.2383 ms	| Score = 0.8839096911507552	| Rel Speedup = 2.0x	| fast_log_loss_e2e
        Time = 0.1235 ms	| Score = 0.8839096911507552	| Rel Speedup = 98.8x	| fast_log_loss
    Benchmarking log_loss... (num_samples=100000, num_classes=10, num_repeats=10
        Time = 26.0730 ms	| Score = 2.587538848622231	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 17.5600 ms	| Score = 2.587538848622231	| Rel Speedup = 1.5x	| ag_log_loss
        Time = 6.3358 ms	| Score = 2.587538848622231	| Rel Speedup = 4.1x	| fast_log_loss_e2e
        Time = 0.1276 ms	| Score = 2.587538848622231	| Rel Speedup = 204.3x	| fast_log_loss
    Benchmarking log_loss... (num_samples=100000, num_classes=100, num_repeats=10
        Time = 150.9778 ms	| Score = 4.915104829696929	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 136.9180 ms	| Score = 4.915104829696929	| Rel Speedup = 1.1x	| ag_log_loss
        Time = 6.6452 ms	| Score = 4.915104829696929	| Rel Speedup = 22.7x	| fast_log_loss_e2e
        Time = 0.1232 ms	| Score = 4.915104829696929	| Rel Speedup = 1225.6x	| fast_log_loss
    Benchmarking log_loss... (num_samples=1000000, num_classes=2, num_repeats=3
        Time = 146.9060 ms	| Score = 0.8855251809037331	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 142.4049 ms	| Score = 0.8855251809037331	| Rel Speedup = 1.0x	| ag_log_loss
        Time = 67.7100 ms	| Score = 0.8855251809037331	| Rel Speedup = 2.2x	| fast_log_loss_e2e
        Time = 1.1628 ms	| Score = 0.8855251809037331	| Rel Speedup = 126.3x	| fast_log_loss
    Benchmarking log_loss... (num_samples=1000000, num_classes=10, num_repeats=3
        Time = 304.8170 ms	| Score = 2.593063906209183	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 208.0698 ms	| Score = 2.593063906209183	| Rel Speedup = 1.5x	| ag_log_loss
        Time = 73.5778 ms	| Score = 2.593063906209183	| Rel Speedup = 4.1x	| fast_log_loss_e2e
        Time = 1.2427 ms	| Score = 2.593063906209183	| Rel Speedup = 245.3x	| fast_log_loss
    Benchmarking log_loss... (num_samples=1000000, num_classes=100, num_repeats=3
        Time = 1507.9544 ms	| Score = 4.911096768402581	| Rel Speedup = 1.0x	| sk_log_loss
        Time = 1354.8307 ms	| Score = 4.911096768402581	| Rel Speedup = 1.1x	| ag_log_loss
        Time = 79.2647 ms	| Score = 4.911096768402581	| Rel Speedup = 19.0x	| fast_log_loss_e2e
        Time = 1.2035 ms	| Score = 4.911096768402581	| Rel Speedup = 1252.9x	| fast_log_loss
    """

    for num_samples, num_classes, num_repeats in [
        (100, 2, 1000),
        (1000, 2, 1000),
        (1000, 10, 1000),
        (1000, 100, 1000),
        (10000, 2, 100),
        (10000, 10, 100),
        (10000, 100, 100),
        (100000, 2, 10),
        (100000, 10, 10),
        (100000, 100, 10),
        (1000000, 2, 3),
        (1000000, 10, 3),
        (1000000, 100, 3),
    ]:
        benchmark_log_loss(num_samples=num_samples, num_classes=num_classes, num_repeats=num_repeats)
