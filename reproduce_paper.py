
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from models import FedANN, FedConvNN
from load_data import X_train, y_train, X_test, y_test
from partition import iid_partition, noniid_partition


# Experiment arguments
parser = argparse.ArgumentParser()

parser.add_argument("--model", choices=["ann", "cnn"], default="cnn")
parser.add_argument("--mode", choices=["iid", "noniid"], default="noniid")
parser.add_argument("--selected", type=int, default=10)
parser.add_argument("--rounds", type=int, default=300)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()

N_CLIENTS = 100
BATCH_SIZE = 128
LR = 0.0003

assert args.selected in [1, 2, 3, 4, 5, 10]
assert args.rounds > 0
assert args.epochs > 0

config = {
    "model": args.model,
    "mode": args.mode,
    "selected": args.selected,
    "rounds": args.rounds,
    "epochs": args.epochs,
    "seed": args.seed
}

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

Model = FedANN if args.model == "ann" else FedConvNN

# Create separate output directory for each experiment
run_name = (
    f"{args.model}_{args.mode}"
    f"_c{args.selected}"
    f"_e{args.epochs}"
    f"_r{args.rounds}"
    f"_seed{args.seed}"
)

out = Path("results") / run_name
out.mkdir(parents=True, exist_ok=True)

checkpoint_path = out / "latest.pt"

# Client data partitions
if args.mode == "iid":
    clients = iid_partition(y_train, N_CLIENTS, args.seed)
else:
    clients = noniid_partition(y_train, N_CLIENTS, args.seed)

# Convert data once
Xtr = torch.tensor(X_train, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.long)
Xte = torch.tensor(X_test, dtype=torch.float32)


# Client-side training
def train_client(global_model, ids):
    model = Model()
    model.load_state_dict(global_model.state_dict())

    dataset = TensorDataset(
        Xtr[ids],
        ytr[ids]
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999)
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for _ in range(args.epochs):
        for X, y in loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    weights = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
    }

    return weights, len(ids)


# Weighted FedAvg aggregation
def aggregate(updates):
    total = sum(n for _, n in updates)
    result = {}

    for key in updates[0][0]:
        avg = torch.zeros_like(updates[0][0][key])

        for weights, n in updates:
            avg += weights[key] * (n / total)

        result[key] = avg

    return result


# Global model evaluation
def evaluate(model):
    model.eval()

    with torch.no_grad():
        output = model(Xte)
        preds = torch.argmax(output, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    return acc, f1, preds


def save_history(history):
    np.savetxt(
        out / "history.csv",
        np.array(history),
        delimiter=",",
        header="round,accuracy,f1",
        comments=""
    )


# Initialize global model
global_model = Model()
history = []
start_round = 0

# Resume an interrupted experiment
if args.resume:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if ckpt["config"] != config:
        raise ValueError("Checkpoint configuration mismatch")

    global_model.load_state_dict(ckpt["model"])
    history = ckpt["history"]
    start_round = ckpt["round"]

    # Restore random states for reproducibility
    torch.set_rng_state(ckpt["torch_rng"])
    rng.bit_generator.state = ckpt["numpy_rng"]

    print("Resuming from round:", start_round)

elif checkpoint_path.exists():
    raise RuntimeError(
        "This experiment already has a checkpoint. "
        "Use --resume or choose a different run."
    )


print("\n========== EXPERIMENT ==========")
print("Model:", args.model)
print("Distribution:", args.mode)
print("Selected clients:", args.selected)
print("Local epochs:", args.epochs)
print("Total rounds:", args.rounds)
print("================================")


# Federated learning
for r in range(start_round, args.rounds):

    selected = rng.choice(
        N_CLIENTS,
        size=args.selected,
        replace=False
    )

    updates = []

    for client_id in selected:
        ids = clients[int(client_id)]

        weights, n = train_client(global_model, ids)
        updates.append((weights, n))

    new_weights = aggregate(updates)
    global_model.load_state_dict(new_weights)

    acc, f1, preds = evaluate(global_model)
    history.append([r + 1, acc, f1])

    print(
        f"Round {r+1:3d}/{args.rounds} | "
        f"Accuracy: {acc*100:.2f}% | "
        f"Macro F1: {f1:.4f}",
        flush=True
    )

    # Save reproducible training state every 25 rounds
    if (r + 1) % 25 == 0 or (r + 1) == args.rounds:

        ckpt = {
            "round": r + 1,
            "model": global_model.state_dict(),
            "history": history,
            "config": config,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": rng.bit_generator.state
        }

        torch.save(ckpt, checkpoint_path)
        save_history(history)

    # Save paper checkpoint models
    if (r + 1) % 50 == 0:
        torch.save(
            global_model.state_dict(),
            out / f"model_round{r+1}.pth"
        )
        print(f"Saved checkpoint at round {r+1}")


# Final evaluation
acc, f1, preds = evaluate(global_model)

names = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

report = classification_report(
    y_test,
    preds,
    labels=list(range(6)),
    target_names=names,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    preds,
    labels=list(range(6))
)

(out / "classification_report.txt").write_text(report)

np.savetxt(
    out / "confusion_matrix.csv",
    cm,
    fmt="%d",
    delimiter=","
)

torch.save(
    global_model.state_dict(),
    out / "final_model.pth"
)

print("\nFinal accuracy:", f"{acc*100:.2f}%")
print("Final macro F1:", f"{f1:.4f}")
print("\n", report)
print("Experiment completed!")
