import torch
import copy
import time
from src.config import config

class Trainer:
    """
    Classe gérant le cycle d'entraînement et de validation.
    """
    def __init__(self, model, loaders, criterion, optimizer):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = config.DEVICE
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def train(self, epochs=10):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
                
                # Itération sur les batchs
                for inputs, labels in self.loaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        # Backward + Optimize
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            
                    # Stats
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                dataset_size = len(self.loaders[phase].dataset)
                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects.double() / dataset_size
                
                self.history[f'{phase}_loss'].append(epoch_loss)
                self.history[f'{phase}_acc'].append(epoch_acc.item())
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Deep Copy du meilleur modèle
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    
            print()
            
        time_elapsed = time.time() - start_time
        print(f'Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Meilleure Accuracy Validation: {best_acc:.4f}')
        
        # Charger les meilleurs poids
        self.model.load_state_dict(best_model_wts)
        return self.model, self.history

    def save_model(self, filename="best_model.pth"):
        path = f"{config.MODEL_SAVE_PATH}/{filename}"
        torch.save(self.model.state_dict(), path)
        print(f"Modèle sauvegardé : {path}")
