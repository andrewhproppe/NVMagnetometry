from pathlib import Path
import matplotlib
import platform

install_path = Path(__file__)
top = install_path.parents[1].absolute()

paths = {
    "raw": top.joinpath("data/raw"),
    "datasets": top.joinpath("data/datasets"),
    "processed": top.joinpath("data/processed"),
    "models": top.joinpath("models"),
    "notebooks": top.joinpath("notebooks"),
    "scripts": top.joinpath("scripts"),
}


def get_system_and_backend():
    if platform.system() == "Linux":
        matplotlib.use("QtAgg")
