import subprocess
import os
from mpi4py import MPI


class Reproducibility:
    def __init__(self, nml):

        if MPI.COMM_WORLD.Get_rank() != 0:
            # Only need to call this from one MPI rank
            return

        output_root = str(nml["meta"]["output_directory"])
        casename = str(nml["meta"]["simname"])
        output_path = self._output_path = os.path.join(output_root, casename)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self._set_commit_hash()
        self._set_branch_name()
        self._set_diff()

        self.dump_to_file(output_path)

    def dump_to_file(self, output_path):

        out = "Git Branch: "
        out += self.branch
        out += "\n"

        out += "Git Commit Hash: " + self.commit
        out += "Git Diff: \n" + self._diff

        with open(
            os.path.join(output_path, "git_info.txt"), "w", encoding="utf8"
        ) as fhandle:
            fhandle.write(out)

    def _set_branch_name(self):

        cmd = "git rev-parse --abbrev-ref HEAD".split()

        self._branch = subprocess.run(
            cmd, capture_output=True, check=True
        ).stdout.decode("utf-8")

    def _set_commit_hash(self):

        cmd = "git rev-parse HEAD".split()
        self._commit = subprocess.run(
            cmd, capture_output=True, check=True
        ).stdout.decode("utf-8")

    def _set_diff(self):
        cmd = "git diff".split()
        self._diff = subprocess.run(cmd, capture_output=True, check=True).stdout.decode(
            "utf-8"
        )

        self._master_commit_hash = self._set_commit_hash

        if self._branch != "master":
            cmd = "git rev-parse origin/master".split()
            master_hash = subprocess.run(
                cmd, capture_output=True, check=True
            ).stdout.decode("utf-8")

            cmd = "git diff master..".split()

            self._diff += (
                "\n On a branch from master: \n so running: git diff master...\n"
            )

            self._diff += "Hash of master is: " + master_hash + "\n"

            self._diff += subprocess.run(
                cmd, capture_output=True, check=True
            ).stdout.decode("utf-8")

    @property
    def branch(self):
        return self._branch

    @property
    def commit(self):
        return self._commit


def main():

    nml = {}
    nml["meta"] = {}
    nml["meta"]["output_directory"] = "./"
    nml["meta"]["simname"] = "test_git"

    Reproducibility(nml)


if __name__ == "__main__":

    main()
