from autobm.utils import GitCheckoutInfo, run_container


def run(podman_client, gitinfo: GitCheckoutInfo, db_base_dir: str, output_dir: str, sql_binary: str, dataset: str, execution_mode: str):
    base_dataset= dataset.split("-")[0]
    return run_container(img="auto-benchmark-runner",
                         bash_cmd=f"""        
        git init lingo-db
        cd lingo-db
        git remote add origin {gitinfo.repo_url}
        git fetch --depth 1 origin {gitinfo.commit_sha}
        git checkout FETCH_HEAD
        cd ..
        python3 run-benchmark.py /db-base {dataset} lingo-db/resources/sql/{base_dataset}/ {execution_mode} /results/results.json
        """,
                         mounts=[{
                             "type": "bind",
                             "source": sql_binary,  # must exist
                             "target": "/usr/bin/sql",
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
                             "source": db_base_dir,  # must exist
                             "target": "/db-base",
                             "read_only": True,  # or True
                             "relabel": "Z",  # SELinux: use "Z" or "z" if needed
                         }],
                         podman_client=podman_client)

# db_base_dir = "/home/michael/projects/auto-benchmark/db-cached/v2/"
# sql_binary = "/home/michael/projects/auto-benchmark/sql"
