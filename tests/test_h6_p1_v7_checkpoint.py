from model.checkpoint_utils import validate_h6_configuration
from model.h6.model import H6Progress1


def _h6():
    return H6Progress1(
        n_groups=1, num_factors=4, bank_dim=8, router_dim=6, text_dim=8, ctx_len=2,
        progress_version="P1-v7-full", expert_enabled=True, expert_bottleneck=2,
    )


def test_p1_v7_checkpoint_metadata_and_strict_h6_load_pass():
    class _Model:
        h6_enabled = True

        def __init__(self):
            self.h6 = _h6()

    model = _Model()
    checkpoint = {
        "checkpoint_version": 7,
        "h6_enabled": True,
        "phase4_progress": 1,
        "h6_config": model.h6.config_dict(),
    }
    validate_h6_configuration(model, checkpoint)
    clone = _h6()
    clone.load_state_dict(model.h6.state_dict(), strict=True)
