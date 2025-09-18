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


# Retrieve all open pull requests
open_prs = repo.get_pulls(state="open")


def post_progress_comment(pr, req_comment, benchmark_config:BenchmarkConfig):
    return pr.create_issue_comment(
        f"""⏳ Benchmark in progress for commit `{pr.head.sha}` (requested by @{req_comment.user.login}). 
        I'll report results here shortly.
        Config:
        * backend: `{benchmark_config.execution_mode}`
        * datasets: {', '.join(f"`{dataset}`" for dataset in benchmark_config.datasets)}
        """
    )


from typing import Dict, List


def _fmt_pct(factor: float, kind: str) -> str:
    if kind == "speedup":
        pct = (1 - factor) * 100  # factor < 1
        return f"{pct:.2f}% speedup"
    else:
        pct = (factor - 1) * 100  # factor > 1
        return f"{pct:.2f}% slowdown"


def _make_table(rows: List[Dict], kind: str) -> str:
    if not rows:
        return "_None._\n"

    def score(it):
        return abs(1 - float(it.get("factor", 1.0)))

    rows_sorted = sorted(rows, key=score, reverse=True)

    header = "| Query | Before | After | Change |\n|:-----:|:------:|:-----:|:------:|"
    lines = [header]
    for it in rows_sorted:
        q = str(it.get("query", "—"))
        before = str(it.get("before", "—"))
        after = str(it.get("after", "—"))
        f = float(it.get("factor", 1.0))
        change = _fmt_pct(f, kind)
        lines.append(f"| `{q}` | {before} | {after} | {change} |")
    return "\n".join(lines) + "\n"


def _format_dataset(name: str, dataset: Dict) -> str:
    parts = [f"## {name}\n",
             "> Times shown as `before` → `after`. Change is computed from the reported factor.\n"]

    for phase in ("compilation", "execution"):
        section = dataset.get(phase)
        if section is None:
            continue

        slow_rows = section.get("slowdown", []) or []
        fast_rows = section.get("speedup", []) or []

        parts.append(f"### {phase.capitalize()}")
        if not slow_rows and not fast_rows:
            parts.append("_No statistically significant changes._\n")
            continue

        if fast_rows:
            parts.append("**🚀 Speedups**")
            parts.append(_make_table(fast_rows, "speedup"))
        else:
            parts.append("**🚀 Speedups**\n_None._\n")

        if slow_rows:
            parts.append("**🐢 Slowdowns**")
            parts.append(_make_table(slow_rows, "slowdown"))
        else:
            parts.append("**🐢 Slowdowns**\n_None._\n")

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
        chunks.append(_format_dataset(name, results_by_dataset[name]))
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
