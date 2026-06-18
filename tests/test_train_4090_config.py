import argparse
import pathlib
from src.train import load_config

_CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "train_4090.yaml"


def test_train_4090_config_merges_over_defaults():
    args = argparse.Namespace(config=str(_CFG))
    cfg = load_config(args)
    assert cfg["device"] == "cuda"
    assert cfg["quantize"] == "4bit"
    assert cfg["max_length"] == 1024
    assert cfg["use_queue"] is True
    assert cfg["batch_size"] == 8
    assert cfg["epochs"] == 3
