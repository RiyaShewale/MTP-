
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from models import FedConvNN
from load_data import X_train, y_train, X_test, y_test
from partition import iid_partition, noniid_partition


# Configuration for our initial test
N_CLIENTS = 100
SELECTED = 10
ROUNDS = 100
LOCAL_EPOCHS = 10
BATCH_SIZE = 128
LR = 0.0003

DATA_MODE = "noniid"
torch.manual_seed(42)
rng = np.random.default_rng(42)

# Divide training data among 100 clients
if DATA_MODE == "iid":
    clients = iid_partition(y_train, N_CLIENTS, seed=42)
elif DATA_MODE == "noniid":
    clients = noniid_partition(y_train, N_CLIENTS, seed=42)
else:
    raise ValueError("Invalid data distribution")


# 1. Client-side training
def train_client(global_model, ids):

    # Each client starts with global weights
    model = FedConvNN()
    model.load_state_dict(global_model.state_dict())

    X = torch.tensor(X_train[ids], dtype=torch.float32)
    y = torch.tensor(y_train[ids], dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()

    for epoch in range(LOCAL_EPOCHS):
        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            output = model(X_batch)
            loss = criterion(output, y_batch)

            loss.backward()
            optimizer.step()

    # Return independent copies of learned weights
    weights = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
    }

    return weights, len(ids)


# 2. Server-side FedAvg aggregation
def fedavg(updates):

    total = sum(n for weights, n in updates)
    new_weights = {}

    for key in updates[0][0]:

        avg = torch.zeros_like(updates[0][0][key])

        for weights, n in updates:
            avg += weights[key] * (n / total)

        new_weights[key] = avg

    return new_weights


# 3. Evaluate the global model
def evaluate(model):

    model.eval()

    X = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        output = model(X)
        preds = torch.argmax(output, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    return acc, f1



# 4. Federated training loop
history = []
global_model = FedConvNN()

run_name = (
    f"fedconvnn_{DATA_MODE}"
    f"_clients{SELECTED}"
    f"_epochs{LOCAL_EPOCHS}"
    f"_rounds{ROUNDS}"
)

print("Model: FedConvNN")
print("Data distribution:", DATA_MODE)
print("Selected clients per round:", SELECTED)

for r in range(ROUNDS):

    print(f"\nCommunication round {r + 1}/{ROUNDS}")

    selected = rng.choice(
        N_CLIENTS,
        size=SELECTED,
        replace=False
    )

    print("Selected clients:", selected)

    updates = []

    # Train selected clients
    for client_id in selected:

        ids = clients[int(client_id)]

        weights, n = train_client(global_model, ids)

        updates.append((weights, n))

    # Aggregate local models
    new_weights = fedavg(updates)

    global_model.load_state_dict(new_weights)

    # Evaluate global model
    acc, f1 = evaluate(global_model)

    history.append([r + 1, acc, f1])

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1: {f1:.4f}")

    # Save checkpoints every 25 rounds
    if (r + 1) % 25 == 0:

        torch.save(
            global_model.state_dict(),
            f"{run_name}_round{r+1}.pth"
        )

        np.savetxt(
            f"{run_name}_history.csv",
            np.array(history),
            delimiter=",",
            header="round,accuracy,f1",
            comments=""
        )

        print(f"Checkpoint saved at round {r+1}")


# 5. Save final results
np.savetxt(
    f"{run_name}_history.csv",
    np.array(history),
    delimiter=",",
    header="round,accuracy,f1",
    comments=""
)

torch.save(
    global_model.state_dict(),
    f"{run_name}.pth"
)

print("\nFederated training completed!")
