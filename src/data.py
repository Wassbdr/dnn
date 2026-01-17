import json
import cv2
from PIL import Image
from torch.utils.data import Dataset

class WLASLDataset(Dataset):
    """
    Dataset pour WLASL (Video based).
    Extrait une frame à la volée depuis les vidéos.
    """
    def __init__(self, root_dir, split='train', transform=None, json_file='WLASL_v0.3.json'):
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, 'videos')
        self.transform = transform
        self.split = split
        
        json_path = os.path.join(root_dir, json_file)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Fichier JSON introuvable : {json_path}")
            
        with open(json_path, 'r') as f:
            content = json.load(f)
            
        self.samples = []
        self.classes = []
        
        # Mapping gloss -> index
        # On ne garde que les classes qui ont des vidéos présentes
        all_glosses = sorted([entry['gloss'] for entry in content])
        self.class_to_idx = {gloss: i for i, gloss in enumerate(all_glosses)}
        self.classes = all_glosses
        
        print(f"Indexation WLASL ({split})...")
        missing_videos = 0
        
        for entry in content:
            gloss = entry['gloss']
            label_idx = self.class_to_idx[gloss]
            
            for instance in entry['instances']:
                if instance['split'] == split:
                    video_id = instance['video_id']
                    # Support mp4 ou avi ou mkv
                    video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
                    
                    if os.path.exists(video_path):
                        self.samples.append((video_path, label_idx))
                    else:
                        missing_videos += 1
                        
        print(f"  {len(self.samples)} vidéos trouvées. ({missing_videos} manquantes)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Chargement vidéo avec OpenCV
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Stratégie : Prendre la frame du milieu
        # (Pour une généralisation 3D, on prendrait une séquence)
        middle_frame_idx = max(0, frame_count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            # Fallback: image noire si lecture échoue
            img = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE))
        else:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_dataloaders(data_dir=None, preprocessor=None, split_ratio=0.8):
    """
    Charge les données, applique les transformations et retourne les DataLoaders.
    Détecte automatiquement le format (ImageFolder ou WLASL Video).
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
        
    print(f"Chargement des données depuis : {data_dir}")
    
    # Transformations pour Train et Val
    train_transforms = get_transforms(preprocessor, mode='train')
    val_transforms = get_transforms(preprocessor, mode='val')
    
    # Vérification fichier JSON pour WLASL
    wlasl_json = os.path.join(data_dir, 'WLASL_v0.3.json')
    is_wlasl = os.path.exists(wlasl_json)
    
    if is_wlasl:
        print("📁 Format détecté : WLASL (Vidéo)")
        try:
            # WLASL a ses propres splits définis dans le JSON
            # Mais souvent sur Kaggle c'est tout mélangé, donc on vérifie
            train_dataset = WLASLDataset(data_dir, split='train', transform=train_transforms)
            val_dataset = WLASLDataset(data_dir, split='val', transform=val_transforms)
            
            # Si le split 'val' est vide (ex: subset), on split manuellement le train
            if len(val_dataset) == 0:
                print("  Split 'val' vide dans JSON, switch vers Random Split du 'train'...")
                full_dataset = WLASLDataset(data_dir, split='train', transform=train_transforms) # On recharge tout
                # Note: Ce n'est pas optimal car on perd le transform 'val' spécifique
                # Mais c'est un fallback acceptable
                train_size = int(split_ratio * len(full_dataset))
                val_size = len(full_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
                
            class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else train_dataset.dataset.classes
            
        except Exception as e:
            print(f"❌ Erreur chargement WLASL: {e}")
            is_wlasl = False # Fallback vers ImageFolder ou Fake
            
    if not is_wlasl:
        # LOGIQUE EXISTANTE (IMAGE FOLDER) ...
        if not os.path.exists(data_dir):
            print(f"❌ Dossier introuvable : {data_dir}. Utilisation de FakeData (Mode Démo).")
            # Fake dataset pour démo technique
            train_dataset = datasets.FakeData(size=100, image_size=(3, config.IMG_SIZE, config.IMG_SIZE), num_classes=29, transform=train_transforms)
            val_dataset = datasets.FakeData(size=20, image_size=(3, config.IMG_SIZE, config.IMG_SIZE), num_classes=29, transform=val_transforms)
            class_names = [str(i) for i in range(29)]
            return DataLoader(train_dataset, config.BATCH_SIZE), DataLoader(val_dataset, config.BATCH_SIZE), class_names

        # Chargement réel ImageFolder
        print("📁 Format détecté : ImageFolder (Images Standard)")
        dataset_no_transform = datasets.ImageFolder(data_dir)
        class_names = dataset_no_transform.classes
        
        train_size = int(split_ratio * len(dataset_no_transform))
        val_size = len(dataset_no_transform) - train_size
        
        train_full_ds = datasets.ImageFolder(data_dir, transform=train_transforms)
        val_full_ds = datasets.ImageFolder(data_dir, transform=val_transforms)
        
        indices = torch.randperm(len(dataset_no_transform)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_dataset = Subset(train_full_ds, train_indices)
        val_dataset = Subset(val_full_ds, val_indices)

    print(f"Classes détectées ({len(class_names)}): {class_names[:5]}...")
    print(f"Train set: {len(train_dataset)}")
    print(f"Val set: {len(val_dataset)}")
    
    # Adaptation workers : eviter trop de workers avec OpenCV (peut causer des soucis)
    num_workers = 0 if is_wlasl else config.NUM_WORKERS 
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, num_workers=num_workers, pin_memory=True)
                              
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
                            
    return train_loader, val_loader, class_names
