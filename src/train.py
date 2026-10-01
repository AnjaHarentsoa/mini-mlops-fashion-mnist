import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mlflow
import mlflow.pytorch


# =========================
# 1. Définition du modèle
# =========================

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


# =========================
# 2. Préparation des données
# =========================

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="data/raw",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.FashionMNIST(
    root="data/raw",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# =========================
# 3. Paramètres
# =========================

learning_rate = 0.001
batch_size = 128
epochs = 5


# =========================
# 4. Device
# =========================

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# =========================
# 5. Expérience MLflow
# =========================

mlflow.set_experiment("Fashion-MNIST")


# =========================
# 6. Run MLflow
# =========================

with mlflow.start_run():

    # =========================
    # Enregistrement paramètres
    # =========================

    mlflow.log_param(
        "learning_rate",
        learning_rate
    )

    mlflow.log_param(
        "batch_size",
        batch_size
    )

    mlflow.log_param(
        "epochs",
        epochs
    )


    # =========================
    # Création du modèle
    # =========================

    model = NeuralNetwork().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # =========================
    # Entraînement
    # =========================

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {average_loss:.4f}"
        )


        # =========================
        # ÉTAPE 12
        # Enregistrement de la Loss
        # =========================

        mlflow.log_metric(
            "train_loss",
            average_loss,
            step=epoch
        )


    # =========================
    # Évaluation
    # =========================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()


    accuracy = correct / total

    print(f"Accuracy : {accuracy:.4f}")


    # =========================
    # ÉTAPE 13
    # Enregistrement Accuracy
    # =========================

    mlflow.log_metric(
        "test_accuracy",
        accuracy
    )


    # =========================
    # ÉTAPE 14
    # Enregistrement du modèle
    # =========================

    mlflow.pytorch.log_model(
        model,
        name="model"
    )


print("Entraînement terminé.")

