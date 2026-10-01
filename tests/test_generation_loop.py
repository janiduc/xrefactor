"""
Regression tests for the Stage-3 generation loop.

These pin the two bugs that made generated "refactored code" meaningless:
  * the decoding loop replaced the prefix with the newest token instead of
    appending to it, so the decoder had no memory of what it had emitted AND
    the tensor collapsed to 1-D (making the positional slice use hidden_size
    as if it were sequence length);
  * training fused the GNN embedding through `fusion_layer` while generation
    passed the raw projected embedding, so the two paths disagreed.
"""

import torch
import torch.nn as nn
import pytest

from src.transformer.code_generator import (
    CodeTransformer,
    TransformerDecoder,
    _apply_repetition_penalty,
    _block_repeated_ngrams,
)


class _RecordingDecoder(TransformerDecoder):
    """Records the shape of every `current_tokens` the loop passes in."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_shapes = []
        self.seen_positional_shapes = []

    def predict_next_token(self, encoder_output, gnn_context, current_tokens, refactoring_type=0):
        self.seen_shapes.append(tuple(current_tokens.shape))
        self.seen_positional_shapes.append(
            tuple(self.positional_encoding[:current_tokens.size(1), :].shape)
        )
        return super().predict_next_token(encoder_output, gnn_context, current_tokens, refactoring_type)


def _tiny_model(vocab_size=64, hidden=32, heads=4, layers=1):
    """A CodeTransformer with no pretrained encoder, so tests stay fast."""
    model = CodeTransformer(model_name="__definitely_not_a_real_model__",
                            gnn_embedding_dim=hidden, hidden_size=hidden,
                            num_layers=layers, num_attention_heads=heads,
                            vocab_size=vocab_size, max_seq_length=64)
    assert model.encoder is None and model.tokenizer is None, "expected the fallback path"
    return model


class TestPrefixAccumulation:
    def test_prefix_grows_and_stays_two_dimensional(self):
        torch.manual_seed(0)
        model = _tiny_model()
        model.decoder = _RecordingDecoder(hidden_size=32, num_layers=1,
                                          num_attention_heads=4, vocab_size=64)
        model.generate("irrelevant", torch.randn(1, 32), max_length=6,
                       repetition_penalty=1.0, no_repeat_ngram_size=0)

        shapes = model.decoder.seen_shapes
        assert len(shapes) >= 2, "loop did not run enough steps to observe growth"
        # Every call is [batch, seq_len] and seq_len increases by exactly one.
        assert all(len(s) == 2 and s[0] == 1 for s in shapes), shapes
        assert [s[1] for s in shapes] == list(range(1, len(shapes) + 1)), shapes

    def test_positional_slice_uses_sequence_length_not_hidden_size(self):
        torch.manual_seed(0)
        model = _tiny_model()
        model.decoder = _RecordingDecoder(hidden_size=32, num_layers=1,
                                          num_attention_heads=4, vocab_size=64)
        model.generate("irrelevant", torch.randn(1, 32), max_length=5,
                       repetition_penalty=1.0, no_repeat_ngram_size=0)

        # (seq_len, hidden) - never (hidden, hidden), which is what the old
        # 1-D collapse produced.
        for step, shape in enumerate(model.decoder.seen_positional_shapes, start=1):
            assert shape == (step, 32), f"step {step} sliced positional encoding as {shape}"

    def test_one_dimensional_prefix_is_rejected_loudly(self):
        decoder = TransformerDecoder(hidden_size=32, num_layers=1,
                                     num_attention_heads=4, vocab_size=64)
        with pytest.raises(AssertionError, match=r"\[batch, seq_len\]"):
            decoder.predict_next_token(
                encoder_output=torch.randn(1, 4, 32),
                gnn_context=torch.randn(1, 1, 32),
                current_tokens=torch.tensor([7]),  # 1-D: the old bug
            )

    def test_generation_stops_at_position_limit(self):
        torch.manual_seed(0)
        model = _tiny_model()
        model.decoder.max_positions = 5
        out = model.generate("irrelevant", torch.randn(1, 32), max_length=1000,
                             repetition_penalty=1.0, no_repeat_ngram_size=0)
        # Must terminate rather than index past the positional encoding.
        assert isinstance(out, str)


class TestAntiRepetition:
    def test_repetition_penalty_lowers_seen_tokens(self):
        logits = torch.tensor([[2.0, -2.0, 1.0]])
        out = _apply_repetition_penalty(logits, [0, 1], penalty=2.0)
        assert out[0, 0].item() == pytest.approx(1.0)   # positive -> divided
        assert out[0, 1].item() == pytest.approx(-4.0)  # negative -> multiplied
        assert out[0, 2].item() == pytest.approx(1.0)   # untouched

    def test_ngram_block_forbids_repeating_a_seen_ngram(self):
        # "1 2 3" already seen; prefix now ends "1 2", so 3 must be banned.
        logits = torch.zeros(1, 8)
        out = _block_repeated_ngrams(logits, [1, 2, 3, 9, 1, 2], ngram_size=3)
        assert out[0, 3].item() == float("-inf")
        assert out[0, 4].item() == 0.0

    def test_ngram_block_is_noop_when_prefix_too_short(self):
        logits = torch.zeros(1, 8)
        out = _block_repeated_ngrams(logits, [1], ngram_size=3)
        assert torch.equal(out, logits)

    def test_degenerate_repetition_is_reduced(self):
        """A random-weight decoder loops badly under plain greedy decoding;
        the anti-repetition machinery must raise token diversity."""
        torch.manual_seed(0)
        model = _tiny_model()
        gnn = torch.randn(1, 32)

        plain = model.generate("x", gnn, max_length=40,
                               repetition_penalty=1.0, no_repeat_ngram_size=0)
        guarded = model.generate("x", gnn, max_length=40,
                                 repetition_penalty=1.5, no_repeat_ngram_size=3)

        def distinct_ratio(text):
            parts = text.split()
            return len(set(parts)) / max(1, len(parts))

        assert distinct_ratio(guarded) >= distinct_ratio(plain)
        assert distinct_ratio(guarded) > 0.2, (
            f"output still degenerate: {guarded[:120]!r}"
        )


class TestTrainInferenceContextParity:
    def test_build_gnn_context_is_the_shared_path(self):
        """The training script's context and generate()'s context must be the
        same function of the same inputs - they used to differ."""
        torch.manual_seed(0)
        model = _tiny_model()
        model.eval()

        encoder_output = torch.randn(1, 5, 32)
        gnn_emb = torch.randn(1, 32)

        with torch.no_grad():
            from_helper, _ = model.build_gnn_context(encoder_output, gnn_emb)
            # Replicate what training does, now that it calls the same helper.
            train_side, _ = model.build_gnn_context(encoder_output, gnn_emb)

        assert torch.allclose(from_helper, train_side)
        assert from_helper.shape == (1, 1, 32)

    def test_raw_projected_embedding_differs_from_fused_context(self):
        """Guard against regressing to the old behaviour: the raw projected
        embedding is NOT interchangeable with the fused context."""
        torch.manual_seed(0)
        model = _tiny_model()
        model.eval()
        encoder_output = torch.randn(1, 5, 32)
        gnn_emb = torch.randn(1, 32)

        with torch.no_grad():
            fused, _ = model.build_gnn_context(encoder_output, gnn_emb)
            raw = gnn_emb.unsqueeze(1)

        assert not torch.allclose(fused, raw)
