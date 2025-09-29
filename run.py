import math
import re
import time

from github import Github
from datetime import datetime

from autobm import benchmark_pr, BenchmarkException, BenchmarkConfig
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Authenticate with a personal access token (or other credentials)
# Replace 'YOUR_TOKEN' with your token or use environment variables
g = Github(config["github_token"])

# Specify repository
repo = g.get_repo(config["repo"])  # e.g., "octocat/Hello-World"

# Define users allowed to request benchmarks
BENCHMARK_REQUESTERS = set(config["whitelisted_requesters"])
BOT_USER = config["bot_user"]


# Helper: decide if this PR needs benchmarking

def needs_benchmark(pr):
    # Fetch all issue comments (includes comments on the PR)
    comments = list(pr.get_issue_comments())
    # Sort by creation time
    comments.sort(key=lambda c: c.created_at)

    # Find the most recent '/benchmark' request by allowed users
    last_request = None
    for c in comments:
        if c.user.login in BENCHMARK_REQUESTERS and c.body.strip().startswith("/benchmark"):
            last_request = c

    if not last_request:
        return False, None

    # Check for a bot response after the last request
    for c in comments:
        if c.user.login == BOT_USER and c.created_at > last_request.created_at and c.body.strip().startswith("✅"):
            # Bot has answered
            return False, last_request

    # No bot answer after last request => needs benchmarking
    return True, last_request



def post_progress_comment(pr, req_comment, benchmark_config:BenchmarkConfig):
    return pr.create_issue_comment(
        f"""⏳ Benchmark in progress for commit `{pr.head.sha}` (requested by @{req_comment.user.login}). 
        I'll report results here shortly.
        Config:
        * backend: `{benchmark_config.execution_mode}`
        * datasets: {', '.join(f"`{dataset}`" for dataset in benchmark_config.datasets)}
        """
    )


from typing import Dict, List, Tuple, Any, Sequence


def _format_pct(x: Any) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.2f}%"

def _format_range(from_pct: Any, to_pct: Any) -> str:
    return f"{_format_pct(from_pct)} - {_format_pct(to_pct)}"

def _first_present(d: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    for k in keys:
        val = d.get(k)
        if isinstance(val, dict):
            return val
    return {}

def _collect_summary_and_tables(
    queries: Dict[str, Any],
    judgement_keys: Sequence[str],
) -> Tuple[Dict[str, int], List[Tuple[str, float, float, float]], List[Tuple[str, float, float, float]]]:
    """
    Returns:
      summary_counts: {"same": n, "undecided": n, "different": n}
      table_speedups: list for faster == "B": (query_id, from_pct, to_pct, sort_metric)
      table_slowdowns: list for faster == "A": (query_id, from_pct, to_pct, sort_metric)
    sort_metric is max(|from|, |to|) for sorting by largest reported effect first.
    """
    counts = {"same": 0, "undecided": 0, "different": 0}
    speedups: List[Tuple[str, float, float, float]] = []
    slowdowns: List[Tuple[str, float, float, float]] = []

    for qid, qdata in (queries or {}).items():
        j = _first_present(qdata or {}, judgement_keys)
        overall = str(j.get("overall", "")).lower()
        faster = j.get("faster")
        from_pct = j.get("from_pct")
        to_pct = j.get("to_pct")

        if overall in counts:
            counts[overall] += 1

        if overall == "different":
            try:
                f_from = float(from_pct)
                f_to = float(to_pct)
                sort_metric = max(abs(f_from), abs(f_to))
            except (TypeError, ValueError):
                f_from = math.nan
                f_to = math.nan
                sort_metric = -math.inf  # will sort last via key function below

            row = (str(qid), f_from, f_to, sort_metric)
            if faster == "B":
                speedups.append(row)
            elif faster == "A":
                slowdowns.append(row)

    # Sort by largest effect range first; NaNs last
    def sort_key(t):
        metric = t[3]
        return (-(metric) if metric == metric else float("inf"), )  # NaN check: NaN != NaN

    speedups.sort(key=sort_key)
    slowdowns.sort(key=sort_key)

    return counts, speedups, slowdowns

def _render_table(rows: List[Tuple[str, float, float, float]], header: str) -> str:
    if not rows:
        return f"*No entries for {header.lower()}.*\n"
    md = []
    md.append(f"**{header}**")
    md.append("")
    md.append("| Query | Range (%) |")
    md.append("|:-----:|:----------|")
    for qid, f_from, f_to, _ in rows:
        md.append(f"| `{qid}` | {_format_range(f_from, f_to)} |")
    md.append("")
    return "\n".join(md)

def _render_section(title: str, queries: Dict[str, Any], judgement_keys: Sequence[str]) -> str:
    counts, speedups, slowdowns = _collect_summary_and_tables(queries, judgement_keys)
    same = counts.get("same", 0)
    undecided = counts.get("undecided", 0)
    different = counts.get("different", 0)

    md = []
    md.append(f"### {title}")
    md.append("")
    md.append(
        f"**Summary:** {same} queries are *same*, {undecided} are *close*, and {different} are *different*."
    )
    md.append("")
    md.append(_render_table(speedups, "Speedups"))
    md.append(_render_table(slowdowns, "Slowdowns"))
    return "\n".join(md)
def _format_dataset(name: str, dataset: Dict) -> str:
    parts = [f"## {name}\n"]

    for phase in ("compilation", "execution"):
        parts.append(_render_section(phase.title(), dataset, [f"judgement_{phase}"]))
    return "\n".join(parts)

def format_benchmarks_markdown(results_by_dataset: Dict[str, Dict]) -> str:
    """
    results_by_dataset:
      {
        "tpch-1": {
          "execution": {"slowdown": [...], "speedup": [...]},
          "compilation": {"slowdown": [...], "speedup": [...]}
        },
        "tpch-2": { ... },
        ...
      }
    """
    chunks = []
    for name in sorted(results_by_dataset.keys()):
        chunks.append(_format_dataset(name, results_by_dataset[name]["queries"]))
    return "\n".join(chunks)


def post_results_comment(pr, results):
    body = f"""✅ Benchmark completed for commit `{pr.head.sha}`.

{format_benchmarks_markdown(results)}
"""
    return pr.create_issue_comment(body)


def post_error_comment(message):
    return f"❌ Benchmark Error: {message}"


import re
import shlex
from typing import Any, Dict, List


def parse_benchmark_args(s: str, config) -> BenchmarkConfig:
    """
    Parse arguments like:
      --datasets tpch-1,tpcds-1,job --backend SPEED --dry-run
      --backend=SLOW --datasets="tpch-1, tpcds-10"
    Rules:
      - --key value  or  --key=value
      - boolean flags (no value) become True
      - repeated flags accumulate as lists
      - datasets is split on commas; backend keeps the last value
    Returns a dict of options ({} if none).
    """
    tokens = shlex.split(s.replace("\r", " ").replace("\n", " "))

    options: Dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            m = re.match(r"^--([A-Za-z0-9][A-Za-z0-9_-]*)(?:=(.*))?$", t)
            if m:
                key = m.group(1).lower()
                if m.group(2) is not None:
                    val = m.group(2)
                else:
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                        val = tokens[i + 1]
                        i += 1
                    else:
                        val = True  # boolean flag

                if key in options:
                    if not isinstance(options[key], list):
                        options[key] = [options[key]]
                    options[key].append(val)
                else:
                    options[key] = val
        i += 1
    available_datasets = set(config["available_datasets"])
    # Normalize known flags
    if "datasets" in options:
        vals = options["datasets"]
        if not isinstance(vals, list):
            vals = [vals]
        datasets: List[str] = []
        for v in vals:
            if v is True:
                continue
            for part in str(v).split(","):
                part = part.strip()
                if part and part in available_datasets:
                    datasets.append(part)
        options["datasets"] = datasets
    backend = options.get("backend", config["default_execution_mode"]).upper()
    datasets = options.get("datasets", config["default_datasets"])
    return BenchmarkConfig(
        execution_mode=backend,
        datasets=datasets
    )

while True:
    # Retrieve all open pull requests
    open_prs = repo.get_pulls(state="open")

    for pr in open_prs:
        needs, req_comment = needs_benchmark(pr)

        print(f"PR #{pr.number}: {pr.title}")
        print(f"  - Head ref: {pr.head.ref}")
        print(f"  - Head SHA: {pr.head.sha}")
        print(f"  - Clone URL: {pr.head.repo.clone_url}")
        print("Wants to merge into")
        print(f"  - Base ref: {pr.base.ref}")
        print(f"  - Base SHA: {pr.base.sha}")
        print(f"  - Base clone URL: {pr.base.repo.clone_url}")

        # Example git commands to fetch and checkout this PR locally:
        print("  Git commands:")
        print(f"    git fetch origin pull/{pr.number}/head:{pr.head.ref}")
        print(f"    git checkout {pr.head.ref}")
        # print(extract_catalog_version(repo, pr.head.sha))
        # print(extract_container_image(repo, pr.head.sha))
        if needs:
            print(f"  -> Needs benchmark (requested at {req_comment.created_at.isoformat()})\n")
            benchmark_config = parse_benchmark_args(req_comment.body.strip(), config)
            progress_comment = post_progress_comment(pr, req_comment, benchmark_config)
            try:
                result = benchmark_pr(pr, benchmark_config, config)
                post_results_comment(pr, result)
            except BenchmarkException as e:
                post_error_comment(e.message)
            progress_comment.delete()
        else:
            print("  -> No benchmark needed\n")
    time.sleep(30)
