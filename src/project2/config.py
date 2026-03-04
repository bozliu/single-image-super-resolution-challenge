from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _resolve_path(project_root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((project_root / p).resolve())


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    project_root = config_path.parent.parent
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(project_root)

    # Resolve common paths.
    data_cfg = cfg.get("data", {})
    for key in [
        "train_hr_dir",
        "train_lr_dir",
        "val_hr_dir",
        "val_lr_dir",
        "test_lr_dir",
    ]:
        if key in data_cfg:
            data_cfg[key] = _resolve_path(project_root, data_cfg[key])

    run_dirs = cfg.setdefault("run_dirs", {})
    run_dirs["results_dir"] = _resolve_path(project_root, run_dirs.get("results_dir", "results"))
    run_dirs["checkpoints_dir"] = _resolve_path(project_root, run_dirs.get("checkpoints_dir", "checkpoints"))
    run_dirs["outputs_dir"] = _resolve_path(
        project_root,
        run_dirs.get("outputs_dir", "Generated upscaled images from testset"),
    )
    run_dirs["report_dir"] = _resolve_path(project_root, run_dirs.get("report_dir", "report"))

    cfg["data"] = data_cfg
    cfg["run_dirs"] = run_dirs
    return cfg
