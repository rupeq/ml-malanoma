import os

import pretrainedmodels
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn import functional as f
import albumentations as alb
from torch.utils.data import DataLoader
from wtfml.data_loaders.image.loader import ClassificationLoader
from wtfml.engine import Engine
from wtfml.utils import EarlyStopping
from sklearn import metrics
from torch.cuda import amp
from apex import amp


torch.cuda.empty_cache()


os.environ['KMP_DUPLICATE_LIB_OK']='True'


train_folder = "./data/train/prepared_data"
model_folder = "./data/model"
test_folder = "./data/test/prepared_data"
train_folds = "./data/train_folds.csv"
test_folds = "./data/test.csv"
device = "cuda"
epochs = 2
train_bs = 2
valid_bs = train_bs // 2
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
lr = 1e-4


class ResNext(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(ResNext, self).__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](
            pretrained=pretrained
        )
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = f.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        output = self.out(x)
        loss = nn.BCEWithLogitsLoss()(
            output, targets.reshape(-1, 1).type_as(output)
        )
        return output, loss


def train(fold: int):
    df = pd.read_csv(train_folds)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    train_alb = alb.Compose(
        [
            alb.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )
    valid_alb = alb.Compose(
        [
            alb.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )
    train_images = df_train.image_name.values.tolist()
    train_images = [
        os.path.join(train_folder, image + ".jpg") for image in train_images
    ]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [
        os.path.join(train_folder, image + ".jpg") for image in valid_images
    ]
    valid_targets = df_valid.target.values

    train_ds = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_alb
    )
    valid_ds = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_alb
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=1
    )

    model = ResNext().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      patience=3,
      mode="max"
    )
    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level="O1",
        verbosity=0
    )
    es = EarlyStopping(patience=5)
    engine = Engine(model, optimizer, device, fp16=True)
    for epoch in range(epochs):
        training_loss = engine.train(
            train_loader
        )
        predictions, valid_loss = engine.evaluate(
            train_loader
        )
        predictions = np.vstack(predictions).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"epoc={epoch}, auc={auc}")
        es(auc, model, os.path.join(model_folder, f"model{fold}.bin"))
        if es.early_stop:
            print("early stopping")
            break


def predict(fold: int):
    df_test = pd.read_csv(test_folds)
    df_test.loc[:, "target"] = 0
    test_alb = alb.Compose(
        [
            alb.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )
    test_images = df_test.image_name.values.tolist()
    test_images = [
        os.path.join(train_folder, image + ".jpg") for image in test_images
    ]
    test_targets = df_test.target.values
    test_ds = ClassificationLoader(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_alb
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=1
    )

    model = ResNext().to(device)
    model.load_state_dict(
        torch.load(os.path.join(model_folder, f"model{fold}.bin"))
    )
    engine = Engine(model=model, device=device, optimizer=None)
    predictions = engine.predict(test_loader)
    return np.vstack(predictions).ravel()


if __name__ == "__main__":
    train(fold=0)
