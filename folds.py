import os

import pandas as pd
from sklearn import model_selection

path = "./data/"


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(path, "train.csv"))
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold_, (_, _) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, "kfold"] = fold_
    df.to_csv(os.path.join(path, "train_folds.csv"), index=False)
