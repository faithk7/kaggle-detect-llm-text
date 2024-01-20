import neptune
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def get_xgboost_model():
    """Get the XGBoost model"""
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc")
    return model


def get_data():
    """Get the data"""
    DATA_ROOT = "/Users/kaiqu/kaggle-datasets/llm-detect-ai-generated-text"
    train_essays_df = pd.read_csv(f"{DATA_ROOT}/train_essays.csv")
    # Here, replace this with actual data loading
    essays = train_essays_df["text"].tolist()  # Replace with actual texts
    labels = train_essays_df[
        "generated"
    ].tolist()  # Replace with actual labels (0 or 1)
    return essays, labels


def tokenize_texts(essays):
    """Tokenize texts using BERT tokenizer"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # You may need additional preprocessing depending on your data
    encoded_essays = tokenizer(
        essays, padding=True, truncation=True, return_tensors="pt"
    )
    return encoded_essays["input_ids"].numpy()  # Convert to numpy array for XGBoost


def train(X, y):
    """Train the XGBoost model"""
    model = get_xgboost_model()
    model.fit(X, y)
    return model


if __name__ == "__main__":
    essays, labels = get_data()

    run = neptune.init_run(
        name="this is a test",
        project="faithk7/detect-llm-text",
    )

    # Convert essays to format suitable for XGBoost
    X = tokenize_texts(essays)

    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # Train the model
    model = train(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict_proba(X_test)[:, 1]  # Get probability predictions
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"AUC Score: {auc_score}")

    run.stop()
