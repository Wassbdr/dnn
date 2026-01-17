import torch
import torch.nn as nn
from torchvision import models
from src.config import config

class ASLModel(nn.Module):
    """
    CNN personnalisé 'From Scratch'.
    Architecture simple et efficace pour la démonstration académique.
    """
    def __init__(self, num_classes):
        super(ASLModel, self).__init__()
        # Feature Extractor
        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloc 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloc 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calcul dynamique de la dimension aplatie
        # Input: 224 -> 112 -> 56 -> 28
        self.flatten_dim = 128 * (config.IMG_SIZE // 8) * (config.IMG_SIZE // 8)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

def get_model(model_name="custom", num_classes=29):
    """
    Factory pour instancier le modèle souhaité.
    Args:
        model_name (str): "custom" ou "mobilenet_v2"
        num_classes (int): Nombre de classes en sortie
    """
    print(f"Création du modèle : {model_name}")
    
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights='DEFAULT')
        # On remplace la dernière couche classifier
        # MobileNetV2 classifier est un Sequential, le dernier élément est le Linear
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    else:
        # Default to Custom
        model = ASLModel(num_classes)
        
    return model.to(config.DEVICE)
