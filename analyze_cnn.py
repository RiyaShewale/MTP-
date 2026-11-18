
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from models import FedConvNN
from load_data import X_test, y_test

names = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

X = torch.tensor(X_test, dtype=torch.float32)

for mode in ["iid", "noniid"]:

    print(f"\n========== {mode.upper()} ==========")

    model = FedConvNN()

    model.load_state_dict(
        torch.load(
            f"fedconvnn_{mode}.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    with torch.no_grad():
        output = model(X)
        preds = torch.argmax(output, dim=1).numpy()

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        preds,
        labels=list(range(6)),
        target_names=names,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        preds,
        labels=list(range(6))
    ))

    print("\nPredicted class counts:")
    print(np.bincount(preds, minlength=6))
