import pathlib
import subprocess

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_run_sh_syntax_valid():
    r = subprocess.run(["bash", "-n", str(_ROOT / "run.sh")])
    assert r.returncode == 0


def test_run_sh_steps_in_order():
    t = (_ROOT / "run.sh").read_text()
    assert t.index("setup-cuda") < t.index("smoke-cuda") < t.index("train-cuda")


def test_run_sh_requires_nvidia_smi():
    assert "nvidia-smi" in (_ROOT / "run.sh").read_text()


def test_makefile_has_cuda_targets():
    mk = (_ROOT / "Makefile").read_text()
    for tgt in ("setup-cuda:", "smoke-cuda:", "train-cuda:"):
        assert tgt in mk
