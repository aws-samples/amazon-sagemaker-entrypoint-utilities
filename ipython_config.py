import os
import subprocess

####################################################################################################
# Additional PYTHONPATH to allow notebooks to import custom modules at a few pre-defined places.

_cwd = os.getcwd()
_line = 'sys.path.append("{}")'
_pythonpath = [
    "import sys, os",
    _line.format(os.getcwd()),
]

# Add GIT_ROOT/ and a few other subdirs
try:
    _p = subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if _p.returncode == 0:
        _git_root = _p.stdout[:-1].decode("utf-8")  # Remove trailing '\n'
        _pythonpath += [
            _line.format(_git_root),  # GIT_ROOT
            _line.format(os.path.join(_git_root, "src")),  # GIT_ROOT/src
            _line.format(os.path.join(_git_root, "notebooks")),  # GIT_ROOT/notebooks
        ]
except:  # noqa: E722
    pass

c.InteractiveShellApp.exec_lines = _pythonpath  # type: ignore # noqa: F821
####################################################################################################
