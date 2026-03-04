from pathlib import Path

import yaml

from project2.config import load_config


def test_load_config_resolves_project_relative_paths(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = config_dir / "smoke.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "train_hr_dir": "data/Mini-DIV2K/Train/HR",
                    "train_lr_dir": "data/Mini-DIV2K/Train/LR_x4",
                    "val_hr_dir": "data/Mini-DIV2K/Val/HR",
                    "val_lr_dir": "data/Mini-DIV2K/Val/LR_x4",
                    "test_lr_dir": "data/test/LR",
                },
                "run_dirs": {
                    "results_dir": "results",
                    "checkpoints_dir": "checkpoints",
                    "outputs_dir": "Generated upscaled images from testset",
                    "report_dir": "report",
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert Path(cfg["data"]["train_hr_dir"]).is_absolute()
    assert Path(cfg["data"]["train_lr_dir"]).is_absolute()
    assert Path(cfg["run_dirs"]["results_dir"]).is_absolute()
    assert Path(cfg["_project_root"]) == project_root.resolve()
