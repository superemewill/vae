import torch

from model import ModelInput


def test_modelinput_namedtuple_constructible() -> None:
    m = ModelInput(
        user_int_feats=torch.zeros(2, 1, dtype=torch.long),
        item_int_feats=torch.zeros(2, 1, dtype=torch.long),
        user_dense_feats=torch.zeros(2, 0),
        item_dense_feats=torch.zeros(2, 0),
        seq_data={},
        seq_lens={},
        seq_time_buckets={},
    )
    assert m.intent_secondary_multi_hot is None
    assert m.intent_confidence is None
