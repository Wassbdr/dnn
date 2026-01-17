import cv2
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image

class BasePreprocessor(ABC):
    """
    Classe abstraite définissant l'interface pour tous les modules de prétraitement/segmentation.
    Permet de tester différentes stratégies de segmentation (Mains, Visage, etc.) de manière modulaire.
    """
    
    @abstractmethod
    def process(self, image_np):
        """
        Prend une image (numpy array RGB) et retourne l'image traitée (numpy array RGB).
        """
        pass
    
    def __call__(self, image_pil):
        """
        Permet d'utiliser cette classe directement dans les transforms de torchvision.
        Accepte PIL, convertit en Numpy, traite, et renvoie PIL.
        """
        img_np = np.array(image_pil)
        processed_np = self.process(img_np)
        return Image.fromarray(processed_np)


class NoOpPreprocessor(BasePreprocessor):
    """
    Passe-plat : ne fait aucune modification. Sert de baseline.
    """
    def process(self, image_np):
        return image_np


class HSVSkinSegmenter(BasePreprocessor):
    """
    Ségmente la main en utilisant la couleur de la peau dans l'espace HSV.
    C'est une méthode de "Computer Vision Classique".
    
    Ref: Une approche simple mais efficace pour isoler la main du fond.
    """
    def __init__(self, lower_hsv=None, upper_hsv=None):
        # Valeurs par défaut pour la couleur de peau (à ajuster selon éclairage)
        if lower_hsv is None:
            self.lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        else:
            self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
            
        if upper_hsv is None:
            self.upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        else:
            self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)

    def process(self, image_np):
        # Conversion RGB -> HSV
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Création du masque
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Nettoyage du masque (Morphology)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Application du masque sur l'image originale
        # On met le fond en noir
        result = cv2.bitwise_and(image_np, image_np, mask=mask)
        
        return result

# TODO: Future implementations
# class MediaPipeHandSegmenter(BasePreprocessor): ...
