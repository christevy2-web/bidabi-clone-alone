import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

# --- 1. CONFIGURATION DES TRANSFORMATIONS (Data Augmentation) ---
# Ces étapes préparent les images pour ResNet-18
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_model():
    # --- 2. CHARGEMENT DU DATASET ---
    # On pointe précisément vers le dossier des images RAW
    data_dir = os.path.join("data", "raw", "images")
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f" Erreur : Le dossier {data_dir} est vide ou inexistant.")
        print("Avez-vous bien lancé le scrapper et organisé les images par dossier ?")
        return

    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    
    # Séparation 80% Train / 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f" Dataset chargé : {len(full_dataset)} images trouvées.")
    print(f"Classes détectées : {full_dataset.classes}")

    # --- 3. CONFIGURATION DU MODÈLE RESNET-18 ---
    # On utilise un modèle pré-entraîné
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # On adapte la dernière couche au nombre de classes (ex: sugar, milk, bread)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- 4. ENTRAÎNEMENT SIMPLIFIÉ ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print(" Début de l'entraînement...")
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / train_size
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    # --- 5. SAUVEGARDE DU MODÈLE ---
    # On le sauvegarde à la racine du projet pour DVC
    model_path = "best_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f" Entraînement terminé ! Modèle sauvegardé sous : {model_path}")

if __name__ == "__main__":
    train_model()