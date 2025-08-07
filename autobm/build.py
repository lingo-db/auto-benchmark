from autobm.utils import GitCheckoutInfo, run_container


def build_sql(output_dir, gitinfo: GitCheckoutInfo, release_build_container, podman_client):
    return run_container(img=release_build_container,
                         bash_cmd=f"""
        git init lingo-db
        cd lingo-db
        git remote add origin {gitinfo.repo_url}
        git fetch --depth 1 origin {gitinfo.commit_sha}
        git checkout FETCH_HEAD
        mkdir build
        cmake -G Ninja . -B build -DCMAKE_BUILD_TYPE=Release -DClang_DIR=/built-llvm/lib/cmake/clang -DArrow_DIR=/built-arrow/lib64/cmake/Arrow -DENABLE_TESTS=OFF
        cmake --build build -j $(nproc) --target sql
        ls build
        cp build/sql /output/sql
        """,
                         mounts=[{
                             "type": "bind",
                             "source": output_dir,  # must exist
                             "target": "/output",
                             "read_only": False,  # or True
                             "relabel": "Z",  # SELinux: use "Z" or "z" if needed
                         }],
                         podman_client=podman_client)


def build_db(db_output_dir, gitinfo: GitCheckoutInfo, podman_client, sql_binary: str, dataset_dir: str, base_dataset:str):
    return run_container(img="auto-benchmark-runner", bash_cmd=f"""
        git init lingo-db
        cd lingo-db
        git remote add origin {gitinfo.repo_url}
        git fetch --depth 1 origin {gitinfo.commit_sha}
        git checkout FETCH_HEAD
        CURR_DIR=`pwd`
        cd /dataset
        sql /built-db < ${{CURR_DIR}}/resources/sql/{base_dataset}/initialize.sql
        touch /built-db/.built
        """,
                         mounts=[{
                             "type": "bind",
                             "source": sql_binary,  # must exist
                             "target": "/usr/bin/sql",
                             "read_only": True,  # or True
                             "relabel": "Z",  # SELinux: use "Z" or "z" if needed
                         }, {
                             "type": "bind",
                             "source": db_output_dir,  # must exist
                             "target": "/built-db",
                             "read_only": False,  # or True
                             "relabel": "Z",  # SELinux: use "Z" or "z" if needed
                         }, {
                             "type": "bind",
                             "source": dataset_dir,  # must exist
                             "target": "/dataset",
                             "read_only": True,  # or True
                             "relabel": "Z",  # SELinux: use "Z" or "z" if needed
                         }],
                         podman_client=podman_client)

# dataset_dir="/home/michael/projects/auto-benchmark/datasets/tpch-1"
# sql_binary="/home/michael/projects/auto-benchmark/sql"
# base_dataset="tpch"
