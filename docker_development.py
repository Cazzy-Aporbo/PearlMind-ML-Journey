"""Docker and deployment reference. The original mixed-language material is preserved under docs/reference.
Run the documented devcontainer instead of executing this reference as Python.
"""
from pathlib import Path

if __name__ == "__main__":
    print(Path(__file__).parent.joinpath("docs/reference/docker_development.py.txt").read_text())
