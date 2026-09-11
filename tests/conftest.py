import pathlib
import sys

# The GPU host's venv belongs to the prior project; don't install into it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
