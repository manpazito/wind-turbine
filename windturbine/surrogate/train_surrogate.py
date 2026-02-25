from __future__ import annotations

import argparse
from pathlib import Path

from windturbine.config import load_config
from windturbine.rotor.polars import PolarModel
from windturbine.surrogate.dataset import generate_training_dataset
from windturbine.surrogate.model import PolynomialSurrogate


def train_surrogate(config_path: Path, output_model_path: Path) -> PolynomialSurrogate:
    cfg = load_config(config_path)
    polar = PolarModel.from_config(cfg.polar)

    x, y, _ = generate_training_dataset(
        rotor_cfg=cfg.rotor,
        optimizer_cfg=cfg.optimizer,
        polar=polar,
        n_samples=cfg.surrogate.n_samples,
        seed=cfg.surrogate.seed,
    )
    if x.shape[0] < 8:
        raise RuntimeError("Insufficient BEM samples to train surrogate model")

    model = PolynomialSurrogate(degree=cfg.surrogate.degree)
    model.fit(x, y)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train BEM surrogate model")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument(
        "--output-model",
        default="outputs/surrogate_model.json",
        help="Where to save the trained model",
    )
    args = parser.parse_args(argv)

    train_surrogate(Path(args.config), Path(args.output_model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
