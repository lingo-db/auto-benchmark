from dataclasses import dataclass

from podman.errors import ContainerError


@dataclass
class GitCheckoutInfo:
    repo_url :str
    commit_sha: str


def run_container(img : str, bash_cmd: str, mounts, podman_client):
    try:
        podman_client.containers.run(img,
                                     ["/bin/bash", "-e", "-c", bash_cmd],
                                     remove=True,
                                     stderr=True,
                                     mounts=mounts)
        return True
    except ContainerError as e:
        # podman-py already includes stderr in the exception
        print("Could not build lingo-db:")
        print("Exit code:", e.exit_status)
        for error in e.stderr:
            print("Error:", error.decode() if isinstance(error, (bytes, bytearray)) else error)
        return False



@dataclass
class BenchmarkConfig:
    datasets: list[str]
    execution_mode: str


# benchmark exception
class BenchmarkException(Exception):
    """Base class for all benchmark-related exceptions."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message