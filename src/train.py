from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB


def train():
    pass


def train_cnn():
    pass


def train_ml(tf_train, y_train, tf_test, y_test):
    """Train the ML model"""
    clf = MultinomialNB(alpha=0.02)
    #     clf2 = MultinomialNB(alpha=0.01)
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
    p6 = {
        "n_iter": 2500,
        "verbose": -1,
        "objective": "cross_entropy",
        "metric": "auc",
        "learning_rate": 0.00581909898961407,
        "colsample_bytree": 0.78,
        "colsample_bynode": 0.8,
        "lambda_l1": 4.562963348932286,
        "lambda_l2": 2.97485,
        "min_data_in_leaf": 115,
        "max_depth": 23,
        "max_bin": 898,
    }

    lgb = LGBMClassifier(**p6)
    cat = CatBoostClassifier(
        iterations=2000,
        verbose=0,
        l2_leaf_reg=6.6591278779517808,
        learning_rate=0.005599066836106983,
        subsample=0.4,
        allow_const_label=True,
        loss_function="CrossEntropy",
    )

    weights = [0.068, 0.31, 0.31, 0.312]

    ensemble = VotingClassifier(
        estimators=[("mnb", clf), ("sgd", sgd_model), ("lgb", lgb), ("cat", cat)],
        weights=weights,
        voting="soft",
        n_jobs=-1,
    )

    ensemble.fit(tf_train, y_train)
