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
import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

import numpy as np
import scipy.stats as stats
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

@dataclass
class ComparisonResult:
    overall: str # can be one of same, different, undecided
    faster: str=None
    from_pct: float=0.0
    to_pct: float=0.0



def compare_runtimes(
    v1, v2, *,
    effect_threshold=0.02,   # practical significance (e.g., ±5%)
    alpha=0.05,              # 1 - confidence (0.05 -> 95% CI)
    n_boot=5000,
    random_state=0,
    return_message=True
)->ComparisonResult:
    """
    Compare runtimes using a bootstrap CI for the *relative* difference in means.

    Relative difference is defined as:
        rel = (mean(v2) - mean(v1)) / ((mean(v1) + mean(v2)) / 2)
    Interpretation:
        rel > 0  -> Version 1 is faster (v2 has higher runtime)
        rel < 0  -> Version 2 is faster
        rel ~ 0  -> similar

    Decision logic (based on CI vs effect_threshold):
      - hi < -effect_threshold        -> "Version 2 is faster"
      - lo >  +effect_threshold       -> "Version 1 is faster"
      - [-thr, +thr] fully contains CI-> "Both versions are basically the same..."
      - otherwise                     -> "Please add more runs..."

    Returns:
        dict with:
          verdict: str
          point: float                 # relative difference (e.g., -0.08 = 8% faster for v2)
          ci: (lo, hi)                 # CI for relative difference
          mean_v1, mean_v2: float
          n_v1, n_v2: int
          message: str (optional)      # formatted human-readable summary
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    if np.any(~np.isfinite(v1)) or np.any(~np.isfinite(v2)):
        raise ValueError("Inputs must be finite numbers.")
    if len(v1) < 2 or len(v2) < 2:
        return ComparisonResult(overall="undecided")

    def rel_diff(a, b):
        ma, mb = np.mean(a), np.mean(b)
        denom = (ma + mb) / 2.0
        return 0.0 if denom == 0 else (mb - ma) / denom

    point = rel_diff(v1, v2)

    # Bootstrap CI
    rng = np.random.default_rng(random_state)
    n1, n2 = len(v1), len(v2)
    idx1 = rng.integers(0, n1, size=(n_boot, n1))
    idx2 = rng.integers(0, n2, size=(n_boot, n2))

    m1b = v1[idx1].mean(axis=1)
    m2b = v2[idx2].mean(axis=1)
    denom = (m1b + m2b) / 2.0
    safe = denom != 0
    boots = np.empty(n_boot, dtype=float)
    boots[~safe] = 0.0
    boots[safe] = (m2b[safe] - m1b[safe]) / denom[safe]

    lo = float(np.quantile(boots, alpha/2))
    hi = float(np.quantile(boots, 1 - alpha/2))
    lo_pct, hi_pct = lo * 100.0, hi * 100.0  # numeric for comparisons

    # Decision
    # Decision
    if hi < -effect_threshold:
        return ComparisonResult(overall="different", faster="B", from_pct=abs(hi_pct), to_pct=abs(lo_pct))
    elif lo > effect_threshold:
        return ComparisonResult(overall="different", faster="A",from_pct=lo_pct, to_pct=hi_pct)
    elif (-effect_threshold <= lo) and (hi <= effect_threshold):
        verdict = "Both versions are basically the same runtime (except for noise)"
        return ComparisonResult(overall="same")
    else:
        return ComparisonResult(overall="undecided")





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

print(sys.argv)
if len(sys.argv) != 7:
    print("Usage: python run-lingodb.py <db_base_dir> <dataset> <queries_dir> <execution_mode> <output_file>")
    sys.exit(1)

db_base_dir_a = sys.argv[1]
db_base_dir_b = sys.argv[2]
dataset = sys.argv[3]
queries_dir = sys.argv[4]
execution_mode = sys.argv[5]
output_file = sys.argv[6]

class Runner:
    def __init__(self, binary_name, db_base_dir):
        self.binary_name = binary_name
        self.db_base_dir = db_base_dir
    def __enter__(self):
        self.process = Process(f"taskset --cpu-list 1-7 {self.binary_name} {self.db_base_dir}/{dataset}",
                               {"LINGODB_EXECUTION_MODE": execution_mode, "LINGODB_SQL_PROMPT": "0",
                                "LINGODB_SQL_REPORT_TIMES": "1", "LINGODB_PARALLELISM": "7"})
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

def execute_query_on_runner(runner, query):
    for _ in range(3):  # 3 warmup runs
        runner.execute(query)
    execution_times = []
    compilation_times = []
    for i in range(10):
        execution_time, compilation_time, client_total = runner.execute(query)
        # print(f"Run: Execution Time: {execution_time} ms\nCompilation Time: {compilation_time} ms\nClient Total Time: {client_total} ms\n")
        execution_times.append(execution_time)
        compilation_times.append(compilation_time)

    return execution_times, compilation_times
with Runner("sqla", db_base_dir_a) as runnera, Runner("sqlb", db_base_dir_b) as runnerb:
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
        print(query_file)
        execution_times_a = []
        compilation_times_a = []
        execution_times_b = []
        compilation_times_b = []
        executionTimesObservedSpeedup = False
        compilationTimesObservedSpeedup = False
        for i in range(10):
            et_a, ct_a = execute_query_on_runner(runnera, query)
            et_b, ct_b = execute_query_on_runner(runnerb, query)
            execution_times_a+=et_a
            compilation_times_a+=ct_a
            execution_times_b+=et_b
            compilation_times_b+=ct_b
            # Initialize analyzer
            compared_execution_times= compare_runtimes(execution_times_a, execution_times_b)
            compared_compilation_times= compare_runtimes(compilation_times_a, compilation_times_b)
            needs_round = False
            if compared_execution_times.overall=="undecided" or compared_compilation_times.overall=="undecided":
                needs_round = True
            if compared_execution_times=="different" and not executionTimesObservedSpeedup:
                needs_round = True
            if compared_compilation_times=="different" and not compilationTimesObservedSpeedup:
                needs_round = True

            if not needs_round or i == 9:
                results["queries"][query_file.replace(".sql", "")] = {
                    "raw":{
                    "execution_times_a": execution_times_a,
                    "execution_times_b": execution_times_b,
                    "compilation_times_a": compilation_times_a,
                    "compilation_times_b": compilation_times_b,
                    },
                    "judgement_execution": {"overall": compared_execution_times.overall, "faster": compared_execution_times.faster, "from_pct": compared_execution_times.from_pct, "to_pct": compared_execution_times.to_pct},
                    "judgement_compilation": {"overall": compared_compilation_times.overall, "faster": compared_compilation_times.faster, "from_pct": compared_compilation_times.from_pct, "to_pct": compared_compilation_times.to_pct}
                }
                break
execution_same=0
execution_fasterA=0
execution_fasterB=0
execution_undecided=0
compilation_same=0
compilation_fasterA=0
compilation_fasterB=0
compilation_undecided=0
for k,v in results["queries"].items():
    judgement_execution=v["judgement_execution"]
    judgement_compilation=v["judgement_compilation"]
    execution_same+= judgement_execution["overall"]=="same"
    execution_fasterA+= judgement_execution["faster"]=="A"
    execution_fasterB+= judgement_execution["faster"]=="B"
    execution_undecided+= judgement_execution["overall"]=="undecided"
    compilation_same+= judgement_compilation["overall"]=="same"
    compilation_fasterA+= judgement_compilation["faster"]=="A"
    compilation_fasterB+= judgement_compilation["faster"]=="B"
    compilation_undecided+= judgement_compilation["overall"]=="undecided"


print("Execution:")
print("Same", execution_same)
print("A Faster", execution_fasterA)
print("B Faster", execution_fasterA)
print("Undecided", execution_undecided)

print("Compilation:")
print("Same", compilation_same)
print("A Faster", compilation_fasterA)
print("B Faster", compilation_fasterA)
print("Undecided", compilation_undecided)
# write results to file
with open(output_file, 'w') as f:
    json.dump(results, f)
