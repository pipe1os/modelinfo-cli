from modelinfo.architecture import extract_architecture


def test_gpt2_conv1d_projection_uses_output_axis():
    tensors = {"transformer.h.0.attn.c_attn.weight": {"shape": [768, 2304]}}
    assert extract_architecture(tensors) == (1, 768, True)


def test_linear_c_attn_projection_uses_output_axis():
    # Qwen-1 and GPTBigCode name a torch.nn.Linear projection c_attn.
    tensors = {"transformer.h.0.attn.c_attn.weight": {"shape": [12288, 4096]}}
    assert extract_architecture(tensors) == (1, 4096, True)


def test_linear_qkv_projection_retains_output_first_layout():
    tensors = {"model.layers.0.qkv_proj.weight": {"shape": [2304, 768]}}
    assert extract_architecture(tensors) == (1, 768, True)
