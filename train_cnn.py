
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models import FedConvNN
from load_data import X_train, y_train, X_test, y_test


# 1. Set random seed
torch.manual_seed(42)

# 2. Convert NumPy arrays into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# 3. Create datasets
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)


# 4. Create data loaders
train_loader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=128,
    shuffle=False
)


# 5. Initialize model, loss function, optimizer
model = FedConvNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003
)


# 6. Train the model
epochs = 10

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for X, y in train_loader:

        # Forward pass
        output = model(X)

        # Calculate loss
        loss = criterion(output, y)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(train_data)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# 7. Evaluate on test data
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in test_loader:

        output = model(X)
        preds = torch.argmax(output, dim=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())


# 8. Calculate performance metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Macro F1: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    labels=list(range(6)),
    target_names=[
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ],
    zero_division=0
))


# 9. Save trained weights
torch.save(model.state_dict(), "fedconvnn_central.pth")

print("Model saved successfully!")
