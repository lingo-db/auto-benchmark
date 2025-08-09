import os
import re
from tempfile import TemporaryDirectory

from podman import PodmanClient

from autobm import build, benchmark
from autobm.compare import compare_benchmark_results
from autobm.utils import GitCheckoutInfo, BenchmarkConfig, BenchmarkException


# load json from filename
def load_json(filename: str):
    import json
    with open(filename, 'r') as f:
        return json.load(f)


def extract_catalog_version(part) -> int | None:
    """
    Returns the catalog binary version from a line like:
        static constexpr size_t binaryVersion = 1234;
    """
    contents = part.repo.get_contents("include/lingodb/catalog/Catalog.h", ref=part.sha)
    match_catalog_version = re.search(
        r"static\s+constexpr\s+size_t\s+binaryVersion\s*=\s*(\d+)\s*;",
        contents.decoded_content.decode("utf-8")
    )
    return int(match_catalog_version.group(1)) if match_catalog_version else None


def extract_container_image(part) -> str | None:
    """
    Returns the image string from a line like:
        container: ghcr.io/lingo-db/lingodb-py-dev:c26a3f...
    Handles optional quotes, extra spaces, and trailing comments.
    """
    contents = part.repo.get_contents(".github/workflows/build-release.yml", ref=part.sha)
    m = re.search(
        r'(?m)^[ \t]*container[ \t]*:[ \t]*["\']?([^\s"#]+)["\']?',
        contents.decoded_content.decode("utf-8")
    )
    return m.group(1) if m else None


def benchmark_part_with_sql(sql_binary, part, benchmark_config: BenchmarkConfig, config, podman_client):
    git_info = GitCheckoutInfo(repo_url=part.repo.clone_url, commit_sha=part.sha)
    catalog_version = extract_catalog_version(part)
    results_for_datasets = {}
    for dataset in benchmark_config.datasets:
        base_dataset = dataset.split("-")[0]
        data_set_dir = os.path.abspath(f"{config["dataset_dir"]}/{dataset}")
        db_base_dir = os.path.abspath(f"{config["db_cache_dir"]}/v{catalog_version}")
        db_dir = os.path.abspath(f"{config["db_cache_dir"]}/v{catalog_version}/{dataset}")
        # check if db_dir/.built exists
        if not os.path.exists(os.path.join(db_dir, ".built")):
            # remove the directory if it exists
            if os.path.exists(db_dir):
                os.rmdir(db_dir)
            # create the directory
            os.makedirs(db_dir)
            if (not build.build_db(
                    db_output_dir=db_dir,
                    gitinfo=git_info,
                    podman_client=podman_client,
                    sql_binary=sql_binary,
                    dataset_dir=data_set_dir,
                    base_dataset=base_dataset
            )):
                raise BenchmarkException(f"Could not build database for commit {part.sha} and dataset {dataset}")
        print(
            f"successfully built database for catalog version {catalog_version} and dataset {dataset} in {db_dir}")
        with TemporaryDirectory() as result_dir:
            result_dir = os.path.abspath(result_dir)
            if not benchmark.run(
                    podman_client=podman_client,
                    gitinfo=git_info,
                    db_base_dir=db_base_dir,
                    output_dir=result_dir,
                    sql_binary=sql_binary,
                    dataset=dataset,
                    execution_mode=benchmark_config.execution_mode
            ):
                raise BenchmarkException(f"Could not run benchmark for commit {part.sha} and dataset {dataset}")
            print(f"successfully ran benchmark for commit {part.sha} and dataset {dataset} in {result_dir}")
            with open(os.path.join(result_dir, "results.json"), "r") as f:
                results = load_json(f.name)
            results_for_datasets[dataset] = results
    return results_for_datasets


def benchmark_part(part, benchmark_config: BenchmarkConfig, config, podman_client):
    git_info = GitCheckoutInfo(repo_url=part.repo.clone_url, commit_sha=part.sha)
    release_build_container = extract_container_image(part)
    # check if cached_binaries/commit_sha/sql exists
    cached_sql_path = os.path.join(config["binaries_cache_dir"], part.sha, "sql")
    if not os.path.exists(cached_sql_path):
        with TemporaryDirectory() as sql_tmp_dir:
            sql_tmp_dir = os.path.abspath(sql_tmp_dir)
            podman_client.images.pull(release_build_container)
            if (not build.build_sql(
                    output_dir=sql_tmp_dir,
                    gitinfo=git_info,
                    release_build_container=release_build_container,
                    podman_client=podman_client,
            )):
                raise BenchmarkException(f"Could not build SQL binary for commit {part.sha}")
            # cache the SQL binary
            os.makedirs(os.path.dirname(cached_sql_path), exist_ok=True)
            os.rename(os.path.join(sql_tmp_dir, "sql"), cached_sql_path)
    return benchmark_part_with_sql(os.path.abspath(cached_sql_path), part, benchmark_config, config, podman_client)


def benchmark_pr(pr, benchmark_config: BenchmarkConfig, config):
    with PodmanClient(base_url=config["podman_socket"]) as client:
        results_after = benchmark_part(pr.head, benchmark_config, config, client)
        results_before = benchmark_part(pr.base, benchmark_config, config, client)
        results = {}
        for dataset in benchmark_config.datasets:
            compared = compare_benchmark_results(results_before[dataset], results_after[dataset])
            results[dataset] = compared
        return results
