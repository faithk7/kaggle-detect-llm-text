import ast
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

if __name__ == "__main__":
    DATA_ROOT = "/Users/kaiqu/kaggle-datasets/llm-detect-ai-generated-text"
    train_essays_df = pd.read_csv(f"{DATA_ROOT}/train_essays.csv")
    predictions = []
    targets = []

    # iterate through the train_esssays_df and write each text column to a .txt file
    for index, row in train_essays_df.iterrows():
        print("index", index)
        text = row["text"]
        with open(f"{index}.txt", "w") as f:
            f.write(text)
        result = subprocess.run(
            ["zippy", f"{index}.txt"],  # the default setting
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:
            # the first line is the file name, so we need to skip it
            pred = result.stdout.split("\n")[1]
            pred = ast.literal_eval(pred)

            print(pred)
            predictions.append(
                pred[0]
            )  # ? what is the meaning of the second output of pred?
            targets.append(row["generated"])
            os.remove(f"{index}.txt")

        else:
            print("error in running zippy")
            break

    # compare the accuracy of the predication with the ground truth
    # TODO: get a better way to retrieve the prediction
    # TODO: convert confidence to probability for calculating the AOC score
    predictions = [0 if "Human" in pred else 1 for pred in predictions]

    predictions = np.array(predictions)
    targets = np.array(targets)

    fpr, tpr, thresholds = roc_curve(targets, predictions)
    auc_score = auc(fpr, tpr)
    print("evaluation score", auc_score)
