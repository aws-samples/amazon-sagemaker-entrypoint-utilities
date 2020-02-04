import pathlib
import pkg_resources

_pkg_dir: pathlib.Path = pathlib.Path(__file__).resolve().parent

try:
    # Loading installed module, so read the installed version
    __version__ = pkg_resources.require(_pkg_dir.name)[0].version
except pkg_resources.DistributionNotFound:
    # Loading uninstalled module, so try to read version from ../VERSION.
    with open(_pkg_dir / ".." / "VERSION", "r") as f:
        __version__ = f.readline().strip()
