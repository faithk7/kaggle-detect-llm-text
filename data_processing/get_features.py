import pandas as pd
import tqdm
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast


def get_data(train_df, test_df):
    y_train, y_test = train_df["generated"], test_df["generated"]
    train_features = pd.DataFrame()
    test_features = pd.DataFrame()

    # get the tf-idf features
    train_df_idf, test_df_idf = get_tf_idf_features(train_df, test_df)

    # TODO: get more features

    train_features = pd.concat([train_features, train_df_idf], axis=1)
    test_features = pd.concat([test_features, test_df_idf], axis=1)

    assert y_train.shape[0] == train_features.shape[0]
    assert y_test.shape[0] == test_features.shape[0]

    return train_df_idf, test_df_idf, y_train, y_test


def get_tf_idf_features(train_df, test_df):
    def get_original_text(text):
        return text

    tokenized_texts_train, tokenized_texts_test = _tokenize_text(train_df, test_df)

    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        analyzer="word",
        tokenizer=get_original_text,
        preprocessor=get_original_text,
        token_pattern=None,
        strip_accents="unicode",
    )

    vectorizer.fit(tokenized_texts_test + tokenized_texts_train)

    # Getting vocab
    vocab = vectorizer.vocabulary_
    print(vocab)

    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        vocabulary=vocab,
        analyzer="word",
        tokenizer=get_original_text,
        preprocessor=get_original_text,
        token_pattern=None,
        strip_accents="unicode",
    )

    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    tf_test = vectorizer.transform(tokenized_texts_test)

    return tf_train, tf_test


def _tokenize_text(train, test):
    LOWERCASE = False
    VOCAB_SIZE = 50522  # original value: 30522

    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    raw_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else []
    )
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

    combined_text_df = train[["text"]]
    combined_text_df = pd.concat([combined_text_df, test[["text"]]])

    dataset = Dataset.from_pandas(combined_text_df)

    def train_corp_iter():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenized_texts_test = []

    for text in tqdm(test["text"].tolist()):
        tokenized_texts_test.append(tokenizer.tokenize(text))

    tokenized_texts_train = []

    for text in tqdm(train["text"].tolist()):
        tokenized_texts_train.append(tokenizer.tokenize(text))

    return tokenized_texts_train, tokenized_texts_test
