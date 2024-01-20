import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


class SklearnCompatibleCNN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_classes,
        filter_sizes,
        num_filters,
        learning_rate=0.001,
        batch_size=32,
        epochs=5,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = TextCNN(
            vocab_size, embed_dim, num_classes, filter_sizes, num_filters
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y):
        dataset = TextDataset(X, y, tokenizer, max_len=512)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
                inputs, labels = batch["text"], batch["labels"]
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for text in X:
                inputs = tokenizer.encode_plus(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                )["input_ids"]
                output = self.model(inputs)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted.numpy())
        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (f, embed_dim)) for f in filter_sizes]
        )
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        x = [
            torch.relu(conv(x)).squeeze(3) for conv in self.convs
        ]  # Apply CNN and ReLU
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # Apply max pooling
        x = torch.cat(x, 1)
        x = self.fc(x)
        return x


# Example model initialization
# model = TextCNN(vocab_size=VOCAB_SIZE, embed_dim=128, num_classes=2, filter_sizes=[3,4,5], num_filters=100)
