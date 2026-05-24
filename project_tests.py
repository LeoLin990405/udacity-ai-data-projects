import numpy as np


def test_tokenize(tokenize):
    tokenized, tokenizer = tokenize(["hello world", "hello machine"])
    assert hasattr(tokenizer, "word_index")
    assert tokenized == [[1, 2], [1, 3]]


def test_pad(pad):
    padded = pad([[1, 2, 3], [4]], length=4)
    assert isinstance(padded, np.ndarray)
    assert padded.shape == (2, 4)
    assert padded.tolist() == [[1, 2, 3, 0], [4, 0, 0, 0]]


def _assert_compiled_sequence_model(model, output_sequence_length, french_vocab_size):
    assert model is not None
    assert model.output_shape[-2] == output_sequence_length
    assert model.output_shape[-1] == french_vocab_size
    assert model.loss is not None


def test_simple_model(simple_model):
    model = simple_model((4, 5, 1), 7, 11, 13)
    _assert_compiled_sequence_model(model, 7, 13)


def test_embed_model(embed_model):
    model = embed_model((4, 5), 7, 11, 13)
    _assert_compiled_sequence_model(model, 7, 13)


def test_bd_model(bd_model):
    model = bd_model((4, 7, 1), 7, 11, 13)
    _assert_compiled_sequence_model(model, 7, 13)


def test_encdec_model(encdec_model):
    model = encdec_model((4, 5, 1), 7, 11, 13)
    _assert_compiled_sequence_model(model, 7, 13)


def test_model_final(model_final):
    model = model_final((4, 5), 7, 11, 13)
    _assert_compiled_sequence_model(model, 7, 13)
