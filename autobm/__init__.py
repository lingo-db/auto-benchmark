import os
import re
from tempfile import TemporaryDirectory

from podman import PodmanClient

from autobm import build, benchmark
from autobm.compare import compare_benchmark_results
from autobm.utils import GitCheckoutInfo, BenchmarkConfig, BenchmarkException, RunConfig


# load json from filename
def load_json(filename: str):
    import json
    with open(filename, 'r') as f:
        return json.load(f)


def extract_catalog_version(git_info: GitCheckoutInfo) -> int | None:
    """
    Returns the catalog binary version from a line like:
        static constexpr size_t binaryVersion = 1234;
    """
    contents = git_info.repo.get_contents("include/lingodb/catalog/Catalog.h", ref=git_info.commit_sha)
    match_catalog_version = re.search(
        r"static\s+constexpr\s+size_t\s+binaryVersion\s*=\s*(\d+)\s*;",
        contents.decoded_content.decode("utf-8")
    )
    return int(match_catalog_version.group(1)) if match_catalog_version else None


def extract_container_image(git_info: GitCheckoutInfo) -> str | None:
    """
    Returns the image string from a line like:
        container: ghcr.io/lingo-db/lingodb-py-dev:c26a3f...
    Handles optional quotes, extra spaces, and trailing comments.
    """
    contents = git_info.repo.get_contents(".github/workflows/build-release.yml", ref=git_info.commit_sha)
    m = re.search(
        r'(?m)^[ \t]*container[ \t]*:[ \t]*["\']?([^\s"#]+)["\']?',
        contents.decoded_content.decode("utf-8")
    )
    return m.group(1) if m else None


def run_benchmark(dataset, run_config_a: RunConfig, run_config_b: RunConfig, execution_mode, gitinfo: GitCheckoutInfo,
                  podman_client):
    with TemporaryDirectory() as result_dir:
        result_dir = os.path.abspath(result_dir)
        if not benchmark.run(
                podman_client=podman_client,
                gitinfo=gitinfo,
                run_config_a=run_config_a,
                run_config_b=run_config_b,
                output_dir=result_dir,
                dataset=dataset,
                execution_mode=execution_mode
        ):
            raise BenchmarkException(f"Could not run benchmark fordataset {dataset}")
        with open(os.path.join(result_dir, "results.json"), "r") as f:
            results = load_json(f.name)
            return results


def build_dataset(git_info: GitCheckoutInfo, sql_binary, dataset, config, podman_client):
    catalog_version = extract_catalog_version(git_info)
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
            raise BenchmarkException(f"Could not build database for commit {git_info.commit_sha} and dataset {dataset}")
    print(
        f"successfully built database for catalog version {catalog_version} and dataset {dataset} in {db_dir}")
    return db_base_dir


def build_sql_binary(git_info: GitCheckoutInfo, config, podman_client):
    release_build_container = extract_container_image(git_info)
    # check if cached_binaries/commit_sha/sql exists
    cached_sql_path = os.path.join(config["binaries_cache_dir"], git_info.commit_sha, "sql")
    print(f"Trying to build {git_info.commit_sha} in {git_info.repo_url}")
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
                raise BenchmarkException(f"Could not build SQL binary for commit {git_info.commit_sha}")
            # cache the SQL binary
            os.makedirs(os.path.dirname(cached_sql_path), exist_ok=True)
            os.rename(os.path.join(sql_tmp_dir, "sql"), cached_sql_path)
    return os.path.abspath(cached_sql_path)


def benchmark_pr(pr, benchmark_config: BenchmarkConfig, config):
    with PodmanClient(base_url=config["podman_socket"]) as client:
        git_info_before = GitCheckoutInfo(repo=pr.base.repo, repo_url=pr.base.repo.clone_url, commit_sha=pr.base.sha)
        git_info_after = GitCheckoutInfo(repo=pr.head.repo, repo_url=pr.head.repo.clone_url, commit_sha=pr.head.sha)
        sql_binary_a = build_sql_binary(git_info_before, config, podman_client=client)
        sql_binary_b = build_sql_binary(git_info_after, config, podman_client=client)
        results = {}
        for dataset in benchmark_config.datasets:
            db_base_dir_a = build_dataset(git_info_before, sql_binary_a, dataset, config, podman_client=client)
            db_base_dir_b = build_dataset(git_info_after, sql_binary_b, dataset, config, podman_client=client)
            run_config_a = RunConfig(sql_binary=sql_binary_a, db_base_dir=db_base_dir_a)
            run_config_b = RunConfig(sql_binary=sql_binary_b, db_base_dir=db_base_dir_b)
            results[dataset] = run_benchmark(dataset, run_config_a, run_config_b, benchmark_config.execution_mode, git_info_before, podman_client=client)
        return results
