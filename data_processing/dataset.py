from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from torch.utils.data import Dataset


class TrainTextDataSet(Dataset):
    def __init__(self, trn_features):
        self.trn_features = trn_features


class ValTextDataSet(Dataset):
    def __init__(self, val_features):
        self.val_features = val_features


class TestTextDataSet(Dataset):
    def __init__(self, test_features):
        self.test_features = test_features
