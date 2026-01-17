import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from src.config import config
from src.preprocessing import NoOpPreprocessor

def get_transforms(preprocessor=None, mode='train'):
    """
    Crée le pipeline de transformations, incluant la segmentation optionnelle.
    Args:
        preprocessor (BasePreprocessor): Un objet de segmentation (ou None).
        mode (str): 'train' ou 'val'.
    """
    if preprocessor is None:
        preprocessor = NoOpPreprocessor()
        
    transform_list = []
    
    # 1. Prétraitement personnalisé (ex: Segmentation)
    # Note: Le préprocesseur reçoit une PIL image via __call__
    transform_list.append(preprocessor)
    
    # 2. Redimensionnement
    transform_list.append(transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)))
    
    # 3. Augmentations (Seulement pour train)
    if mode == 'train':
        transform_list.extend([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        
    # 4. Conversion Tensor & Normalisation
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)

def get_dataloaders(data_dir=None, preprocessor=None, split_ratio=0.8):
    """
    Charge les données, applique les transformations et retourne les DataLoaders.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
        
    print(f"Chargement des données depuis : {data_dir}")
    
    # Transformations pour Train et Val
    train_transforms = get_transforms(preprocessor, mode='train')
    val_transforms = get_transforms(preprocessor, mode='val')
    
    # Vérification dossier
    if not os.path.exists(data_dir):
        print(f"❌ Dossier introuvable : {data_dir}. Utilisation de FakeData (Mode Démo).")
        # Fake dataset pour démo technique
        train_dataset = datasets.FakeData(size=100, image_size=(3, config.IMG_SIZE, config.IMG_SIZE), num_classes=29, transform=train_transforms)
        val_dataset = datasets.FakeData(size=20, image_size=(3, config.IMG_SIZE, config.IMG_SIZE), num_classes=29, transform=val_transforms)
        class_names = [str(i) for i in range(29)]
        return DataLoader(train_dataset, config.BATCH_SIZE), DataLoader(val_dataset, config.BATCH_SIZE), class_names

    # Chargement réel
    # Astuce : On utilise ici ImageFolder "basique" pour découvrir les classes
    # Puis on créera des Subsets avec des transforms spécifiques
    dataset_no_transform = datasets.ImageFolder(data_dir)
    class_names = dataset_no_transform.classes
    
    train_size = int(split_ratio * len(dataset_no_transform))
    val_size = len(dataset_no_transform) - train_size
    
    # Pour appliquer des transforms différents, on instancie deux ImageFolders
    train_full_ds = datasets.ImageFolder(data_dir, transform=train_transforms)
    val_full_ds = datasets.ImageFolder(data_dir, transform=val_transforms)
    
    # Split aléatoire des indices
    indices = torch.randperm(len(dataset_no_transform)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(train_full_ds, train_indices)
    val_dataset = Subset(val_full_ds, val_indices)
    
    print(f"Classes détectées ({len(class_names)}): {class_names[:5]}...")
    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(val_dataset)} images")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
                              
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
                            
    return train_loader, val_loader, class_names
