import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mlflow
import mlflow.pytorch


# ============================================================
# 1. PARAMÈTRES DE L'EXPÉRIENCE
# ============================================================

learning_rate = 0.001
batch_size = 64
epochs = 5
hidden_neurons = 256

# ============================================================
# 2. MODÈLE
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self, hidden_neurons=128):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, hidden_neurons),
            nn.ReLU(),

            nn.Linear(hidden_neurons, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


# ============================================================
# 3. PRÉPARATION DES DONNÉES
# ============================================================

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
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# 4. DEVICE
# ============================================================

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# 5. EXPÉRIENCE MLFLOW
# ============================================================

mlflow.set_experiment("Fashion-MNIST")


# ============================================================
# 6. RUN MLFLOW
# ============================================================

with mlflow.start_run():

    # --------------------------------------------------------
    # Enregistrement des paramètres
    # --------------------------------------------------------

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

    mlflow.log_param(
        "hidden_neurons",
        hidden_neurons
    )


    # --------------------------------------------------------
    # Création du modèle
    # --------------------------------------------------------

    model = NeuralNetwork(
        hidden_neurons=hidden_neurons
    ).to(device)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # ========================================================
    # 7. ENTRAÎNEMENT
    # ========================================================

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for images, labels in train_loader:

            # Envoyer les données sur le device
            images = images.to(device)
            labels = labels.to(device)

            # Réinitialiser les gradients
            optimizer.zero_grad()

            # Prédiction
            outputs = model(images)

            # Calcul de la loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Mise à jour des poids
            optimizer.step()

            # Accumuler la loss
            total_loss += loss.item()


        # Loss moyenne de l'epoch
        average_loss = total_loss / len(train_loader)


        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {average_loss:.4f}"
        )


        # ----------------------------------------------------
        # ÉTAPE 12 — Enregistrement de la Loss
        # ----------------------------------------------------

        mlflow.log_metric(
            "train_loss",
            average_loss,
            step=epoch
        )


    # ========================================================
    # 8. ÉVALUATION
    # ========================================================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Prédiction
            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            # Nombre total d'images
            total += labels.size(0)

            # Nombre de bonnes prédictions
            correct += (
                predictions == labels
            ).sum().item()


    # Calcul de l'accuracy
    accuracy = correct / total


    print(
        f"Accuracy : {accuracy:.4f}"
    )


    # --------------------------------------------------------
    # ÉTAPE 13 — Enregistrement de l'Accuracy
    # --------------------------------------------------------

    mlflow.log_metric(
        "test_accuracy",
        accuracy
    )


    # ========================================================
    # 9. ENREGISTREMENT DU MODÈLE
    # ========================================================

    # Exemple d'entrée nécessaire pour MLflow
    example_input = torch.randn(
        1,
        1,
        28,
        28
    ).to(device)


    # --------------------------------------------------------
    # ÉTAPE 14 — Enregistrement du modèle
    # --------------------------------------------------------

    mlflow.pytorch.log_model(
        model,
        name="model",
        input_example=example_input
    )


    # ========================================================
    # 10. INFORMATIONS DU RUN
    # ========================================================

    run = mlflow.active_run()

    print()
    print("====================================")
    print("MLflow Run terminé")
    print("Run ID:", run.info.run_id)
    print("Accuracy:", accuracy)
    print("====================================")


print()
print("Entraînement terminé.")
