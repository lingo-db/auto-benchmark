from __future__ import annotations

import json
import math
import re
import sys
import time
import os

import selectors
import subprocess
import time

from dataclasses import dataclass
from typing import List, Tuple
import random
import math
import statistics as stats

@dataclass
class StabilityResult:
    stable: bool
    estimate_ms: float | None           # median runtime (ms) if stable, else None
    plus_minus_ms: float | None         # CI half-width (ms) for reporting "x ms ± y ms"
    ci_95: Tuple[float, float] | None   # (low, high) 95% CI for the median
    n_input: int                        # number of samples provided
    n_used: int                         # after outlier trimming
    removed_outliers: int
    cv_pct: float | None                # coefficient of variation (%) on trimmed data
    mad_ms: float | None                # median absolute deviation (ms)
    reason: str                         # short explanation

def _iqr_fences(xs: List[float], k: float = 1.5) -> Tuple[float, float]:
    q1, q3 = _percentile(xs, 25), _percentile(xs, 75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)

def _percentile(xs: List[float], p: float) -> float:
    # linear interpolation, inclusive method; no numpy dependency
    if not xs:
        raise ValueError("empty")
    ys = sorted(xs)
    r = (p/100) * (len(ys)-1)
    i = int(math.floor(r))
    j = int(math.ceil(r))
    if i == j:
        return ys[i]
    frac = r - i
    return ys[i] * (1-frac) + ys[j] * frac

def _bootstrap_ci_median(xs: List[float], alpha: float = 0.05, B: int = 10_000, seed: int | None = 12345) -> Tuple[float,float]:
    rng = random.Random(seed)
    n = len(xs)
    meds = []
    for _ in range(B):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        meds.append(stats.median(sample))
    meds.sort()
    lo_idx = int((alpha/2) * B)
    hi_idx = int((1 - alpha/2) * B) - 1
    return (meds[lo_idx], meds[hi_idx])

def judge_stability(
    times_ms: List[float],
    *,
    min_runs: int = 6,
    rel_tol: float = 0.02,         # 2% relative half-width target
    abs_tol_ms: float = 0.10,      # or ≤ 0.10 ms absolute half-width
    fence_k: float = 1.5,          # Tukey fence multiplier
    alpha: float = 0.05,
    bootstrap_B: int = 10_000,
    bootstrap_seed: int | None = 12345
) -> StabilityResult:
    """Return whether runtimes are stable enough and a robust estimate if so."""
    xs = [float(x) for x in times_ms if math.isfinite(x) and x > 0]
    n_input = len(xs)
    if n_input < 2:
        return StabilityResult(False, None, None, None, n_input, n_input, 0, None, None,
                               "Need ≥2 positive samples")

    # Trim extreme outliers (helps on noisy hosts)
    lo, hi = _iqr_fences(xs, fence_k)
    trimmed = [x for x in xs if lo <= x <= hi]
    removed = n_input - len(trimmed)
    if len(trimmed) < 2:
        # if trimming nuked everything, fall back to raw
        trimmed = xs
        removed = 0

    n = len(trimmed)
    if n < min_runs:
        return StabilityResult(False, None, None, None, n_input, n, removed, None, None,
                               f"Need ≥{min_runs} runs after trimming; have {n}")

    median = stats.median(trimmed)
    mad = stats.median([abs(x - median) for x in trimmed])  # robust spread in ms

    # For diagnostics only: CV on trimmed data (uses mean/std, not for decision)
    mean = stats.fmean(trimmed)
    stdev = stats.pstdev(trimmed) if n > 1 else 0.0
    cv_pct = (stdev / mean * 100.0) if mean > 0 else None

    # Bootstrap 95% CI for the median
    ci_lo, ci_hi = _bootstrap_ci_median(trimmed, alpha=alpha, B=bootstrap_B, seed=bootstrap_seed)
    half_width = (ci_hi - ci_lo) / 2.0

    # Decision: accept if CI half-width is small enough relative to estimate or absolutely
    rel_ok = (median > 0) and (half_width / median <= rel_tol)
    abs_ok = (half_width <= abs_tol_ms)

    if rel_ok or abs_ok:
        return StabilityResult(True, median, half_width, (ci_lo, ci_hi), n_input, n, removed, cv_pct, mad,
                               f"Stable: half-width {half_width:.4f} ms (rel {half_width/median*100:.2f}%); "
                               f"n={n}, outliers removed={removed}")
    else:
        # Give a concrete suggestion for more runs: estimate needed n scaling for width ~ 1/sqrt(n)
        # Bootstrap CI width tends to shrink ~sqrt(n); propose multiplier to hit target
        target_half = max(rel_tol * median, abs_tol_ms)
        suggested_n = math.ceil(n * (half_width / target_half) ** 2)
        suggested_n = max(suggested_n, n + 2)  # at least a few more
        return StabilityResult(False, None, None, (ci_lo, ci_hi), n_input, n, removed, cv_pct, mad,
                               f"Not stable: half-width {half_width:.4f} ms (rel {half_width/median*100:.2f}%). "
                               f"Try ~{suggested_n} total runs.")




class Process:
    def __init__(self, command: str, env: dict = None, cwd: str = None):
        self._command = command
        self._env = env
        self._cwd = cwd

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        #logger.log_verbose_process(f'Starting command `{self._command}`')
        self.process = subprocess.Popen(self._command.split(" "), env=self._env, cwd=self._cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.selector.register(self.process.stderr, selectors.EVENT_READ)

    def stop(self):
        self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, self._command)
        #logger.log_verbose_process(f'Stopped command `{self._command}`')

    def kill(self):
        self.process.stdin.close()
        self.process.terminate()
        #logger.log_verbose_process(f'Killed command `{self._command}`')

    def write(self, text: str):
        self.read_and_discard()

        self.process.stdin.write(text.encode())
        self.process.stdin.write(b"\n")
        self.process.stdin.flush()

        #logger.log_verbose_process(f'stdin:  {text}')

    def wait(self):
        while True:
            for key, _ in self.selector.select():
                data = key.fileobj.readline().decode().strip()

                if data is None:
                    return
                # Check if the process has terminated
                if self.process.poll() is not None:
                    return

    def readline_stderr(self):
        while True:
            for key, _ in self.selector.select():
                data = key.fileobj.readline().decode().strip()

                if data is None:
                    raise ChildProcessError("process closed")
                elif data:
                        return data

                # Check if the process has terminated
                if self.process.poll() is not None:
                    raise ChildProcessError("process closed")

    def read_and_discard(self):
        for key, _ in self.selector.select(timeout=0):
            # Check if the process has terminated
            if self.process.poll() is not None:
                raise ChildProcessError("process closed")

            data = key.fileobj.readline().decode().strip()

    def run(self) -> str:
        #logger.log_verbose_process(f'Running command `{self._command}`')
        ret = subprocess.run(self._command.split(" "), env=self._env, cwd=self._cwd, capture_output=True, text=True)
        if ret.returncode:
            #logger.log_verbose_process_stderr(ret.stderr)
            raise ChildProcessError(ret.stderr)

        return ret.stdout


if len(sys.argv) != 6:
    print("Usage: python run-lingodb.py <db_base_dir> <dataset> <queries_dir> <execution_mode> <output_file>")
    sys.exit(1)

db_base_dir = sys.argv[1]
dataset = sys.argv[2]
queries_dir = sys.argv[3]
execution_mode = sys.argv[4]
output_file = sys.argv[5]

class Runner:
    def __enter__(self):
        self.process = Process(f"taskset --cpu-list 0-7 sql {db_base_dir}/{dataset}",
                               {"LINGODB_EXECUTION_MODE": execution_mode, "LINGODB_SQL_PROMPT": "0",
                                "LINGODB_SQL_REPORT_TIMES": "1", "LINGODB_PARALLELISM": "8"})
        self.process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.stop()

    def execute(self, query: str):

        begin = time.time()
        self.process.write(str(query))
        output: str = None
        client_total = math.nan
        while output is None:
            output = self.process.readline_stderr()
            client_total = (time.time() - begin) * 1000

            if "execution:" in output and "compilation:" in output:
                break
            elif output.startswith("ERROR:") or output.startswith("what():"):
                raise RuntimeError(f"Error executing query: {output}")

            output = None

        try:
            [compilation, execution] = re.findall(r'([0-9.]*) \[ms]', output)
            execution = float(execution)
            compilation = float(compilation)
        except ValueError as e:
            print(f"Could not extract execution and compilation time from '{output}'")
            execution = math.nan
            compilation = math.nan
        return execution, compilation, client_total


results={}
results["dataset"] = dataset
results["queries"] = {}
with Runner() as runner:
    for query_file in os.listdir(queries_dir):
        if "initialize" in query_file:
            continue
        # read query from file
        with open(queries_dir + "/" + query_file, 'r') as f:
            query = f.read()

            # if query is not yet terminated with a semicolon, add it
        if not query.strip().endswith(';'):
            query += ';'
        query += "\n"  # ensure a newline at the end
        for _ in range(3):  # 3 warmup runs
            runner.execute(query)
        execution_times = []
        compilation_times = []
        for i in range(200):
            execution_time, compilation_time, client_total = runner.execute(query)
            # print(f"Run: Execution Time: {execution_time} ms\nCompilation Time: {compilation_time} ms\nClient Total Time: {client_total} ms\n")
            execution_times.append(execution_time)
            compilation_times.append(compilation_time)
            judged_execution_times = judge_stability(
                execution_times,
                rel_tol=0.02,
            )
            judged_compilation_times = judge_stability(
                compilation_times,
                rel_tol=0.02,
            )
            if judged_execution_times.stable and judged_compilation_times.stable:
                results["queries"][query_file.replace(".sql","")] = {
                    "execution_times": execution_times,
                    "compilation_times": compilation_times,
                    "execution_time_summary": f"{judged_execution_times.estimate_ms:.3f} ms\n± {judged_execution_times.plus_minus_ms:.3f} ms",
                    "compilation_time_summary":  f"{judged_compilation_times.estimate_ms:.3f} ms\n± {judged_compilation_times.plus_minus_ms:.3f} ms",
                }
                break

# write results to file
with open(output_file, 'w') as f:
    json.dump(results, f)
