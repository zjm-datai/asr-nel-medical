from __future__ import annotations

from types import SimpleNamespace

import torch

from asr_nec_model.data.collate import collate_label, collate_retrieval
from asr_nec_model.data.types import EntitySpeech, LabelExample, RetrievalExample
from asr_nec_model.inference.pipeline import apply_correction, correct_one_utterance, retrieve_candidates
from asr_nec_model.models.labeler import GenerativeLabeler
from asr_nec_model.utils.tensors import pad_1d


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 9
        self._mapping = {}

    def add_tokens(self, tokens, special_tokens=True):
        return len(tokens)

    def __call__(self, text, add_special_tokens=False, truncation=True, max_length=448, return_tensors="pt"):
        ids = self._mapping[text]
        return SimpleNamespace(input_ids=torch.tensor([ids[:max_length]], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(x)) for x in ids.tolist())


class FakeSearcher:
    def eval(self):
        return self

    def to(self, device):
        return self

    def score(self, utterance_features, entity_features):
        return entity_features[:, 0, 0]


class FakeLabeler:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def generate_label(self, input_features, candidates, asr_text, max_new_tokens=32):
        self.calls.append((tuple(candidates), asr_text))
        if len(candidates) == 1:
            return "foo" if candidates[0] == "B" else "bar"
        return "foo"


def build_stub_labeler():
    labeler = GenerativeLabeler.__new__(GenerativeLabeler)
    tokenizer = FakeTokenizer()
    prompt = "a ||| b <EC> asr"
    tokenizer._mapping[prompt] = [10, 11, 12]
    tokenizer._mapping["target"] = [20, 21]
    tokenizer._mapping["foo"] = [30]
    tokenizer._mapping["bar"] = [31]
    labeler.processor = SimpleNamespace(tokenizer=tokenizer)
    labeler.model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=1))
    return labeler


def test_pad_1d_and_collate_shapes():
    seqs = [torch.tensor([1, 2]), torch.tensor([3])]
    padded = pad_1d(seqs, 0)
    assert padded.shape == (2, 2)
    assert padded.tolist() == [[1, 2], [3, 0]]

    batch = [
        (torch.ones(2, 3), torch.ones(2, 3), torch.tensor(1.0)),
        (torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor(0.0)),
    ]
    utterance, entity, label = collate_retrieval(batch)
    assert utterance.shape == (2, 2, 3)
    assert entity.shape == (2, 2, 3)
    assert label.shape == (2,)


def test_build_decoder_sequence_masks_prompt_tokens():
    labeler = build_stub_labeler()
    prompt = "a ||| b <EC> asr"
    decoder_ids, labels = GenerativeLabeler.build_decoder_sequence(labeler, prompt, "target")
    assert decoder_ids.tolist() == [1, 10, 11, 12, 20, 21]
    assert labels.tolist() == [-100, -100, -100, 20, 21, 9]


def test_retrieve_and_correct_pipeline():
    searcher = FakeSearcher()
    labeler = FakeLabeler()
    utterance = torch.zeros(2, 3)
    datastore = [
        EntitySpeech("A", torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        EntitySpeech("B", torch.tensor([[0.8, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        EntitySpeech("C", torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])),
    ]

    candidates = retrieve_candidates(searcher, utterance, datastore, threshold=0.3, top_k=2, device="cpu")
    assert [name for name, _ in candidates] == ["B", "A"]
    assert abs(candidates[0][1] - 0.8) < 1e-6
    assert abs(candidates[1][1] - 0.4) < 1e-6

    corrected = correct_one_utterance(searcher, labeler, utterance, "foo and bar", datastore[:2], threshold=0.3, top_k=2, device="cpu")
    assert corrected == "B and A"


def test_apply_correction_empty_and_single():
    assert apply_correction("foo", ["A"], "<empty>") == "foo"
    assert apply_correction("foo", ["A"], "foo") == "A"
    assert apply_correction("foo", ["A", "B"], "foo") == "foo"
