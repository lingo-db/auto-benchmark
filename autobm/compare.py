from typing import Iterable, Optional, Dict, Union
import numpy as np, random

Result = Optional[Dict[str, Union[str, float]]]


def significant_speed_change_b_vs_a(
        a: Iterable[float],
        b: Iterable[float],
        *,
        alpha: float = 0.05,
        k: float = 1.5,
        n_boot: int = 10_000,
        n_perm: int = 10_000,
        random_state: Optional[int] = 42,
) -> Result:
    """
    Return {"effect":"speedup"/"slowdown","factor":>1} if significant, else None.
    factor = median(A)/median(B); >1 means B is faster (speedup).
    decision:
      - "ci": significant if bootstrap CI for factor excludes 1 (default)
      - "both": CI excludes 1 AND permutation p<alpha
    """
    if random_state is not None:
        np.random.seed(random_state);
        random.seed(random_state)

    def _clean(xs):
        xs = np.asarray(list(xs), dtype=float)
        return xs[np.isfinite(xs)]

    def _rm_outliers(xs, kk=1.5):
        if xs.size == 0: return xs
        q1, q3 = np.percentile(xs, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - kk * iqr, q3 + kk * iqr
        return xs[(xs >= lo) & (xs <= hi)]

    a = _rm_outliers(_clean(a), k)
    b = _rm_outliers(_clean(b), k)
    if a.size == 0 or b.size == 0:
        raise ValueError("After cleaning/outlier removal, one sample is empty.")

    ma, mb = np.median(a), np.median(b)
    if mb <= 0:
        raise ValueError("Median of B must be positive.")
    factor = float(ma / mb)

    # Bootstrap CI for factor
    na, nb = a.size, b.size
    boot = []
    for _ in range(n_boot):
        sa = np.random.choice(a, size=na, replace=True)
        sb = np.random.choice(b, size=nb, replace=True)
        msb = np.median(sb)
        if msb <= 0: continue
        boot.append(np.median(sa) / msb)
    if not boot: return None
    boot = np.asarray(boot)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if not ((hi < 1.0) or (lo > 1.0)):
        return None

    if factor > 1.0:
        return {"effect": "speedup", "factor": float(factor)}
    else:
        return {"effect": "slowdown", "factor": float(1.0 / factor)}







def compare_detail(query_data_before, query_data_after, type, compare_result, query):
    before = query_data_before[f"{type}_times"]
    after = query_data_after[f"{type}_times"]
    res = significant_speed_change_b_vs_a(before, after, alpha=0.05, k=1.5, n_boot=10_000, n_perm=10_000,
                                          random_state=42)

    if res:
        compare_result[type][res["effect"]].append(
            {"factor": res["factor"], "query": query, "before": query_data_before[f"{type}_time_summary"],
             "after": query_data_after[f"{type}_time_summary"]})


def compare_benchmark_results(results_before, results_after):
    compare_result = {}
    compare_result["compilation"] = {"slowdown": [], "speedup": []}
    compare_result["execution"] = {"slowdown": [], "speedup": []}
    for query in results_before["queries"]:
        if query not in results_after["queries"]:
            continue
        query_data_before = results_before["queries"][query]
        query_data_after = results_after["queries"][query]
        compare_detail(query_data_before, query_data_after, "execution", compare_result, query)
        compare_detail(query_data_before, query_data_after, "compilation", compare_result, query)
    return compare_result
