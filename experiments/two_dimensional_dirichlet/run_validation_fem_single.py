from __future__ import annotations

import argparse
from pathlib import Path

from dirichlet_p1_fem import solve_fem
from dirichlet_validation_patches import config_from_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mesh", type=int, required=True)
    args = parser.parse_args()

    cfg = config_from_json(Path(args.config_json))
    cfg.device = "cpu"
    solve_fem(
        cfg,
        args.mesh,
        Path(args.reference_dir),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
