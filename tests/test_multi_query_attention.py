import pytest

from modelinfo.architecture import extract_architecture


@pytest.mark.parametrize(
    "config,expected",
    [
        # bigcode/gpt_bigcode-santacoder: one shared KV head of size 2048 / 16.
        (
            {"model_type": "gpt_bigcode", "n_layer": 24, "n_head": 16,
             "n_embd": 2048, "multi_query": True},
            (24, 128, False),
        ),
        # tiiuae/falcon-7b: multi-query attention with 71 query heads of size 64.
        (
            {"model_type": "falcon", "num_hidden_layers": 32,
             "num_attention_heads": 71, "hidden_size": 4544,
             "multi_query": True, "new_decoder_architecture": False},
            (32, 64, False),
        ),
        # tiiuae/falcon-40b: grouped-query attention declared as num_kv_heads.
        (
            {"model_type": "falcon", "num_hidden_layers": 60,
             "num_attention_heads": 128, "num_kv_heads": 8, "hidden_size": 8192,
             "new_decoder_architecture": True},
            (60, 512, False),
        ),
        # tiiuae/falcon-rw-1b: multi_query disabled keeps one KV head per query head.
        (
            {"model_type": "falcon", "num_hidden_layers": 24,
             "num_attention_heads": 32, "hidden_size": 2048, "multi_query": False},
            (24, 2048, False),
        ),
    ],
)
def test_kv_heads_follow_multi_query_and_num_kv_heads(config, expected):
    assert extract_architecture({}, config) == expected


def test_explicit_num_key_value_heads_takes_precedence_over_multi_query():
    config = {"num_hidden_layers": 2, "num_attention_heads": 8,
              "num_key_value_heads": 2, "hidden_size": 512, "multi_query": True}
    assert extract_architecture({}, config) == (2, 128, False)
