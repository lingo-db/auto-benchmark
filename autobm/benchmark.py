from autobm.utils import GitCheckoutInfo, run_container, RunConfig


def run(podman_client, gitinfo: GitCheckoutInfo, run_config_a: RunConfig, run_config_b : RunConfig, output_dir: str, dataset: str, execution_mode: str):
    base_dataset= dataset.split("-")[0]
    return run_container(img="auto-benchmark-runner",
                         bash_cmd=f"""
        git init lingo-db
        cd lingo-db
        git remote add origin {gitinfo.repo_url}
        git fetch --depth 1 origin {gitinfo.commit_sha}
        git checkout FETCH_HEAD
        cd ..
        taskset --cpu-list 8-23 python3 run-benchmark.py /db-base-a /db-base-b {dataset} lingo-db/resources/sql/{base_dataset}/ {execution_mode} /results/results.json
        """,
                         mounts=[{
                             "type": "bind",
                             "source": run_config_a.sql_binary,  # must exist
                             "target": "/usr/bin/sqla",
                             "read_only": True,  # or True
                             "relabel": "Z",  # SELinux: use "Z" or "z" if needed
                         },{
                             "type": "bind",
                             "source": run_config_b.sql_binary,  # must exist
                             "target": "/usr/bin/sqlb",
                             "read_only": True,  # or True
                             "relabel": "Z",  # SELinux: use "Z" or "z" if needed
                         }, {
                             "type": "bind",
                             "source": output_dir,  # must exist
                             "target": "/results",
                             "read_only": False,  # or True
                             "relabel": "Z",  # SELinux: use "Z" or "z" if needed
                         }, {
                             "type": "bind",
                             "source": run_config_a.db_base_dir,  # must exist
                             "target": "/db-base-a",
                             "read_only": True,  # or True
                             "relabel": "Z",  # SELinux: use "Z" or "z" if needed
                         }, {
                             "type": "bind",
                             "source": run_config_b.db_base_dir,  # must exist
                             "target": "/db-base-b",
                             "read_only": True,  # or True
                             "relabel": "Z",  # SELinux: use "Z" or "z" if needed
                         }],
                         podman_client=podman_client)

# db_base_dir = "/home/michael/projects/auto-benchmark/db-cached/v2/"
# sql_binary = "/home/michael/projects/auto-benchmark/sql"
