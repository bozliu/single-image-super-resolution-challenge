from project2.model import build_model
from project2.utils import count_parameters


def test_model_stays_within_parameter_budget() -> None:
    cfg = {
        "model": {
            "scale": 4,
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 20,
        }
    }
    model = build_model(cfg)
    params = count_parameters(model)

    assert params == 1812995
    assert params < 1821085
