import os
import torch

class Config:
    def __init__(self):
        # Détection de l'environnement
        self.IS_KAGGLE = os.path.exists("/kaggle/input")
        try:
            import google.colab
            self.IS_COLAB = True
        except ImportError:
            self.IS_COLAB = False
            
        # Définition du Device
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparamètres par défaut
        self.IMG_SIZE = 224
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001
        self.EPOCHS = 10
        self.NUM_WORKERS = 2 if (self.IS_KAGGLE or self.IS_COLAB) else 4
        
        # Chemins des données
        self.DATA_DIR = self._get_data_dir()
        self.MODEL_SAVE_PATH = self._get_save_path()
        
    def _get_data_dir(self):
        """Détermine le chemin des données selon l'environnement."""
        if self.IS_KAGGLE:
            # Chemins potentiels sur Kaggle (à adapter si le dataset change de nom)
            candidates = [
                "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train",
                "/kaggle/input/asl-alphabet-test/asl_alphabet_train/asl_alphabet_train"
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
            return "/kaggle/input/fallback_data_dir"
            
        elif self.IS_COLAB:
            # Sur Colab, on suppose généralement un unzip dans ./data
            return "./data/asl_alphabet_train/asl_alphabet_train"
            
        else:
            # Local
            return "./data/asl_alphabet_train/asl_alphabet_train"

    def _get_save_path(self):
        """Détermine où sauvegarder les modèles."""
        if self.IS_KAGGLE:
            return "/kaggle/working"
        else:
            return "."

    def __str__(self):
        return (f"Config:\n"
                f"  Device: {self.DEVICE}\n"
                f"  Environment: {'Kaggle' if self.IS_KAGGLE else ('Colab' if self.IS_COLAB else 'Local')}\n"
                f"  Data Dir: {self.DATA_DIR}\n"
                f"  Image Size: {self.IMG_SIZE}x{self.IMG_SIZE}")

# Instance globale pour import facile
config = Config()
