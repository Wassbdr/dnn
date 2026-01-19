# Script de Presentation Orale - Pix2Pix : Traduction d'Image a Image

**Duree totale : 15 minutes**

---

# PARTIE 1 : Introduction - Baseline, Infrastructure et Donnees

## Duree cible : 5 minutes

---

## Slide 1 : Titre et contexte
**Duree : 30 secondes**

Bonjour a tous. Aujourd'hui, je vais vous presenter notre travail sur Pix2Pix, une architecture de reseaux antagonistes generatifs conditionnels pour la traduction d'image a image.

Ce projet explore trois domaines d'application distincts :
- La traduction cartographique avec le dataset Maps
- La generation architecturale avec Facades
- La synthese de produits avec Edges2Shoes

Nous avons egalement developpe une application experimentale de synesthesie artificielle, transformant des signaux audio en representations visuelles.

---

## Slide 2 : Architecture Baseline - Vue d'ensemble
**Duree : 1 minute**

L'architecture Pix2Pix, introduite par Isola et al. en 2017, repose sur deux composants fondamentaux.

### Le Generateur U-Net

Cette architecture encoder-decoder possede une propriete essentielle : les **skip-connections**. Ces connexions de saut relient directement chaque couche de l'encodeur a son homologue dans le decodeur.

**Implementation concrete :**
- 8 blocs de downsampling (encodeur)
- 8 blocs d'upsampling (decodeur)
- Dropout a 0.5 dans les 3 premiers blocs du decodeur

Le Dropout introduit de la stochasticite et evite le mode collapse.

### Le Discriminateur PatchGAN

Approche **locale** plutot que globale :
- Evalue des patches de **70x70 pixels** independamment
- Classifie chaque region comme reelle ou fausse
- Force le generateur a produire des textures haute frequence coherentes

**Avantage :** Chaque region locale est scrutee individuellement, garantissant la qualite des details sur l'ensemble de l'image.

---

## Slide 3 : Fonctions de perte
**Duree : 45 secondes**

La fonction de perte combine deux termes complementaires :

### Formule globale
```
Loss_totale = Loss_adversariale + lambda * Loss_L1
```

### Terme 1 : Perte adversariale (BCE)
- Binary Cross-Entropy
- Pousse le generateur a produire des images indiscernables des reelles

### Terme 2 : Perte L1 (Reconstruction)
- Ponderee par **lambda = 100** dans la baseline
- Garantit la coherence structurelle avec la cible
- Empeche les images visuellement realistes mais semantiquement incorrectes

### Point critique
Le choix de **lambda = 100** privilegie fortement la fidelite a la cible, au detriment potentiel du realisme des textures. Ce parametre sera au coeur de nos optimisations.

---

## Slide 4 : Infrastructure technique
**Duree : 45 secondes**

### Environnement d'entrainement
| Composant | Specification |
|-----------|---------------|
| Plateforme | Kaggle / Google Colab |
| GPU | NVIDIA Tesla T4 |
| Memoire VRAM | 16 Go |
| Framework | PyTorch |

### Configuration d'entrainement
| Parametre | Valeur | Justification |
|-----------|--------|---------------|
| Batch Size | 1 | Requis pour Instance Normalization |
| Normalisation | InstanceNorm2d | Calcul par image individuelle |
| Temps/epoque (Maps) | ~1 min 30 sec | 1096 images train |

### Systeme de sauvegarde
- **Checkpointing systematique** : Sauvegarde du meilleur modele
- **Critere** : Validation loss minimale
- **Early stopping** : Patience de 10 epoques

---

## Slide 5 : Dataset et acquisition
**Duree : 1 minute**

### Source unique : Kaggle Pix2Pix Dataset

| Caracteristique | Detail |
|-----------------|--------|
| Source | kaggle.com/datasets/vikramtiwari/pix2pix-dataset |
| Auteur | Vikram Tiwari |
| Taille totale | 2.7 Go |
| Nombre de fichiers | ~56 300 images |
| Format | Paires concatenees horizontalement (512x256) |

### Composition du dataset

Le dataset regroupe 4 sous-ensembles pour la traduction image-a-image :

| Sous-dataset | Tache | Description |
|--------------|-------|-------------|
| **Maps** | Schema -> Satellite | Paires Google Maps (carte schematique / vue satellite) |
| **Facades** | Segmentation -> Photo | Annotations semantiques colorees / facades architecturales |
| **Edges2Shoes** | Contours -> Produit | Croquis de contours / photos de chaussures |
| **Cityscapes** | Segmentation -> Scene | Segmentation urbaine / scene de rue reelle |

### Preprocessing

Chaque image du dataset est une concatenation horizontale :
- **Partie gauche (256x256)** : Image source (input)
- **Partie droite (256x256)** : Image cible (target)

```python
# Separation des paires
input_image = image[:, :, :256]   # Moitie gauche
target_image = image[:, :, 256:]  # Moitie droite
```

**Note :** L'ordre input/target peut varier selon le sous-dataset. Verification visuelle requise.

### Dataset Synesthesia (Generation procedurale)

Pour notre application experimentale de synesthesie, nous avons genere un dataset synthetique :

| Type Audio | Pattern Visuel Associe |
|------------|------------------------|
| Sinusoide | Gradient vertical |
| Chirp | Ondulations |
| Modulation FM | Diagrammes de Voronoi |
| Percussions | Formes radiales |

**Avantage :** Controle total sur les correspondances audio-visuelles.
**Limite :** Generalisation potentiellement limitee aux donnees reelles.

---

## Slide 6 : Problemes identifies dans la baseline
**Duree : 1 minute**

### Probleme 1 : Effondrement du discriminateur

**Observation :**
| Epoque | D-loss |
|--------|--------|
| 40 | ~10^-5 |
| 60 | 1.7 x 10^-6 |

**Diagnostic :** Le discriminateur a appris a distinguer parfaitement les images reelles des fausses.

**Consequence :** Vanishing gradient - le generateur cesse de s'ameliorer car le gradient transmis est quasi-nul.

### Probleme 2 : Flou excessif

**Observation :** Images generees depourvues de details fins, aspect lisse.

**Explication technique :**
- Lambda = 100 donne trop de poids a la perte L1
- Strategie optimale pour minimiser L1 face a l'incertitude : generer la **moyenne des textures possibles**
- Moyenne = flou

### Probleme 3 : Desequilibre D/G

**Metriques de validation :**
- Meilleur G-loss : 27.4 (epoque 49)
- Early stopping : epoque 129
- Ecart croissant entre D-loss et G-loss

**Conclusion :** Systeme piege dans un equilibre sous-optimal.

---

## Transition vers Partie 2

Ces trois problemes ont guide nos experimentations :

1. **Effondrement D** -> Spectral Normalization
2. **Flou excessif** -> Reduction de lambda + VGG Perceptual Loss
3. **Desequilibre D/G** -> LSGAN (MSE) + Learning Rate Scheduling

Passons maintenant a l'analyse critique de nos experimentations sur les trois datasets.

---

# PARTIE 2 : Analyse critique des evolutions
## Duree cible : 8 minutes

---

## Slide 7 : Methodologie experimentale
**Duree : 30 secondes**

Avant de detailler nos resultats, precisons notre demarche scientifique.

### Protocole experimental

Chaque modification suit un cycle rigoureux :

1. **Hypothese** : Identification d'un probleme et proposition d'une solution theorique
2. **Implementation** : Modification isolee d'un seul parametre ou composant
3. **Test** : Entrainement complet avec metriques quantitatives
4. **Analyse** : Comparaison des resultats avec la baseline
5. **Iteration** : Correction ou validation de l'hypothese

### Metriques d'evaluation

| Metrique | Description | Objectif |
|----------|-------------|----------|
| FID | Frechet Inception Distance | Minimiser (qualite perceptuelle) |
| LPIPS | Learned Perceptual Image Patch Similarity | Minimiser (similarite perceptuelle) |
| G-loss | Perte du generateur | Stabiliser |
| D-loss | Perte du discriminateur | Eviter effondrement |

---

## Slide 8 : Facades - Analyse des limites de la baseline
**Duree : 1 minute**

### Contexte experimental

Le dataset Facades presente une tache de generation particulierement exigeante : transformer une segmentation semantique coloree en facade architecturale photorealiste.

Chaque couleur encode un element architectural specifique (fenetre, porte, balcon, mur). Le generateur doit synthetiser des textures realistes tout en respectant strictement la disposition spatiale.

### Observation : Effet peinture

Les facades generees presentent un **effet peinture** caracteristique : les textures manquent de details fins et ressemblent a des aquarelles plutot qu'a des photographies.

### Hypothese

**Cause suspectee :** Lambda = 100 impose une contrainte L1 trop forte.

**Mecanisme :** Face a l'incertitude sur la texture exacte a generer, la strategie optimale pour minimiser L1 est de produire la **moyenne des textures possibles**, resultant en un aspect flou et lisse.

### Test : Augmentation des epoques

| Configuration | Epoques | Observation |
|---------------|---------|-------------|
| Baseline | 100 | Effet peinture present |
| Etendue | 200 | Effet peinture persiste |

### Analyse

L'augmentation du temps d'entrainement n'a pas resolu le probleme fondamental.

**Conclusion :** Le probleme est structurel, lie au ratio de la fonction de perte, non au nombre d'iterations.

### Perspective et transition vers Maps

Cette observation a motive nos experimentations sur Maps et Edges2Shoes : reduction de lambda a 10 combinee a la VGG Perceptual Loss pour privilegier la qualite perceptuelle sur la fidelite pixel-a-pixel.

**Pourquoi Maps ensuite ?** Le dataset Maps presente des defis complementaires a Facades : alors que Facades requiert des textures architecturales fines, Maps necessite une coherence spatiale sur de longues distances (routes, batiments). Les deux partagent le probleme du flou L1, mais Maps ajoute la complexite des structures connectees.

---

## Slide 10 : Maps - Evolution architecturale
**Duree : 1 minute 30 secondes**

### Phase 1 : Baseline (test.ipynb)

**Configuration initiale :**
- U-Net standard + PatchGAN
- BCE + L1 (lambda = 100)
- Batch Normalization
- 200 epoques

**Problemes observes :**
1. Routes discontinues aux intersections
2. Textures repetitives et plates
3. Effondrement du discriminateur (D-loss -> 10^-6)

### Hypothese 1 : Probleme de dependances longue portee

Les routes traversent l'image entiere. Le PatchGAN 70x70 ne peut pas capturer ces structures globales.

**Solution proposee :** Self-Attention

### Implementation Self-Attention

```python
class SelfAttention(nn.Module):
    # Calcul d'attention sur l'ensemble des positions spatiales
    # Permet au generateur de "voir" les connexions distantes
```

**Positionnement :** Apres le 4eme bloc de l'encodeur (resolution 32x32)

**Justification :** A cette resolution, chaque position peut influencer toutes les autres avec un cout computationnel raisonnable.

### Hypothese 2 : Instabilite d'entrainement

**Solution proposee :** Spectral Normalization sur le discriminateur

**Principe :** Contraindre la constante de Lipschitz des couches pour stabiliser les gradients.

**Resultat :** D-loss maintenu dans une plage saine [0.1 - 1.0] au lieu de s'effondrer.

---

## Slide 11 : Maps - Fonctions de perte avancees
**Duree : 1 minute 15 secondes**

### Hypothese 3 : BCE inadaptee

La Binary Cross-Entropy sature rapidement, causant le vanishing gradient.

**Solution proposee :** LSGAN (Least Squares GAN)

### Comparaison theorique

| Aspect | BCE | LSGAN (MSE) |
|--------|-----|-------------|
| Formule D | -log(D(x)) | (D(x) - 1)^2 |
| Gradient loin de 0 | Faible | Fort |
| Penalite echantillons faux | Binaire | Proportionnelle |

**Avantage LSGAN :** Penalise les faux echantillons proportionnellement a leur distance de la cible, meme quand D est confiant.

### Hypothese 4 : L1 seule insuffisante

La perte L1 mesure la difference pixel-a-pixel mais ignore la coherence perceptuelle.

**Solution proposee :** VGG Perceptual Loss

### Implementation VGG Loss

```python
class VGGLoss(nn.Module):
    def __init__(self):
        # VGG16 pre-entraine, features jusqu'a relu3_3
        # Comparaison dans l'espace des features, pas des pixels
```

**Principe :** Comparer les activations intermediaires d'un VGG16 pre-entraine plutot que les pixels bruts.

**Effet :** Le generateur apprend a reproduire des **textures perceptuellement similaires** plutot que des valeurs de pixels identiques.

### Configuration finale Maps (Phase 3)

| Parametre | Valeur |
|-----------|--------|
| L1_LAMBDA | 10 (reduit de 100) |
| VGG_LAMBDA | 10 |
| Loss adversariale | MSE (LSGAN) |
| Normalisation G | InstanceNorm |
| Normalisation D | SpectralNorm |

### Transition vers Edges2Shoes

Les ameliorations validees sur Maps (VGG Loss, LSGAN, Spectral Norm) constituent maintenant notre nouvelle baseline. Mais Edges2Shoes pose un defi different : generer des **textures riches et variees** (cuir, toile, lacets) a partir d'informations minimales (simples contours). 

La question se pose : l'architecture U-Net est-elle optimale pour ce type de tache creative, ou une architecture avec plus de liberte serait-elle preferable ?

---

## Slide 12 : Edges2Shoes - Experimentation ResNet
**Duree : 1 minute 15 secondes**

### Contexte specifique

Fort des ameliorations validees sur Maps, j'ai voulu explorer une piste architecturale differente pour Edges2Shoes.

### Hypothese initiale : U-Net trop contraint ?

Les skip-connections du U-Net forcent une correspondance spatiale stricte entre input et output. Pour Edges2Shoes, ou l'on genere des textures riches a partir de simples contours, une plus grande liberte creative pourrait etre benefique.

**Solution testee :** Generateur ResNet (9 blocs residuels)

### Architecture ResNet Generator

```
Encodeur (3 convolutions) 
    -> 9 blocs residuels (transformation)
    -> Decodeur (3 deconvolutions)
```

**Difference fondamentale :** Pas de skip-connections, transformation dans un espace latent compresse.

### Resultats experimentaux (v3) - Premiers resultats encourageants

| Aspect | U-Net Baseline | ResNet |
|--------|----------------|--------|
| Richesse textures | Moyenne | **Excellente** |
| Variations couleurs | Limitees | **Naturelles** |
| Creativite | Contrainte | **Libre** |

**Observation positive :** Les textures generees etaient remarquablement riches - cuir brillant, toile texturee, lacets detailles. Le ResNet produisait des chaussures visuellement tres convaincantes.

### Probleme decouvert : Brouillard de pixels

Malgre ces resultats prometteurs, un probleme est apparu progressivement : un **brouillard de pixels** (pixel fog) affectait certaines zones de l'image, particulierement dans les regions de transition entre textures.

| Symptome | Description |
|----------|-------------|
| Zones floues | Halos grisatres autour des contours |
| Perte de nettete | Details fins noyes dans le bruit |
| Inconsistance | Qualite variable selon les regions |

**Diagnostic :** Sans les skip-connections, le decodeur perd des informations spatiales fines lors de la reconstruction, generant ce brouillard caracteristique.

### Decision : Retour au U-Net

Malgre la richesse des textures ResNet, le brouillard de pixels degradait trop la qualite finale. **J'ai decide de revenir au U-Net** et d'y integrer les ameliorations validees (VGG Loss, LSGAN) plutot que de changer d'architecture.

**Lecon cle :** Une architecture produisant de bons resultats partiels peut avoir des defauts structurels redhibitoires. Les skip-connections du U-Net sont essentielles pour maintenir la clarte spatiale.

### Transition : Capitaliser sur les acquis

Ce retour au U-Net n'est pas un echec mais un **pivot strategique**. L'experimentation ResNet m'a montre que des textures riches etaient possibles - le probleme etait la methode, pas l'objectif. J'ai donc conserve l'idee d'ameliorer la richesse des textures, mais en utilisant la **VGG Loss** plutot qu'un changement d'architecture. 

L'objectif : obtenir les textures riches du ResNet avec la precision spatiale du U-Net.

---

## Slide 13 : Edges2Shoes - Revolution VGG Loss
**Duree : 1 minute**

### Hypothese

Plutot que de changer d'architecture, pourquoi ne pas changer la fonction de perte ? La VGG Loss devrait ameliorer la qualite perceptuelle sans sacrifier la precision geometrique du U-Net.

### Test : Integration VGG Loss (v4)

**Configuration :**
- U-Net avec skip-connections (conserve)
- VGG Loss ajoutee
- L1_LAMBDA = 10

### Resultats spectaculaires

| Metrique | Baseline (L1 seule) | Avec VGG Loss |
|----------|---------------------|---------------|
| G-loss epoque 2 | ~45 | 7.96 |
| Convergence | 50+ epoques | < 10 epoques |
| Qualite textures | Floue | Nette |

**Observation remarquable :** Record de G-loss = 7.96 des l'epoque 2.

### Analyse

La VGG Loss guide le generateur vers des textures perceptuellement coherentes des les premieres epoques. Le reseau apprend d'abord les caracteristiques haut niveau (structure, texture globale) avant les details fins.

**Comparaison :** Avec L1 seule, le reseau passe des dizaines d'epoques a minimiser des differences pixel-a-pixel souvent non pertinentes perceptuellement.

---

## Slide 14 : Edges2Shoes - Elimination des artefacts
**Duree : 1 minute**

### Probleme persistant : Artefacts en damier (Checkerboard)

**Observation :** Malgre la VGG Loss, certaines images presentent des motifs en damier, particulierement visibles dans les zones unies.

### Diagnostic technique

**Cause identifiee :** ConvTranspose2d (deconvolution)

```python
# Problematique
nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
```

**Explication :** Quand stride ne divise pas kernel_size, certains pixels de sortie recoivent plus de contributions que d'autres, creant un motif periodique.

### Solution : Resize-Convolution (v5)

```python
# Solution
nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
)
```

**Principe :** Separer l'upsampling (interpolation bilineaire) de l'apprentissage des features (convolution standard).

### Resultats

| Version | Artefacts damier | Qualite textures |
|---------|------------------|------------------|
| v4 (ConvTranspose) | Presents | Bonne |
| v5 (Resize-Conv) | Elimines | Excellente |

### Configuration finale Edges2Shoes (v5.1 "Ultimate")

| Composant | Configuration |
|-----------|---------------|
| Generator | U-Net + Resize-Convolution |
| Discriminateur | PatchGAN + Spectral Norm |
| Pertes | MSE + L1 (lambda=10) + VGG |
| Stabilisation | Replay Buffer + TTUR |
| Batch size | 4 (augmente pour stabilite) |
| Early stopping | Patience = 10 |

---

## Slide 15 : Synthese comparative
**Duree : 45 secondes**

### Tableau recapitulatif des evolutions

| Probleme | Dataset principal | Solution | Impact |
|----------|-------------------|----------|--------|
| Inversion I/O | Facades | Verification visuelle | Correction bug critique |
| Flou L1 | Tous | VGG Perceptual Loss | Textures nettes |
| Effondrement D | Maps | Spectral Normalization | Stabilite entrainement |
| Vanishing gradient | Maps, Edges2Shoes | LSGAN (MSE) | Gradients maintenus |
| Routes discontinues | Maps | Self-Attention | Coherence globale |
| Artefacts damier | Edges2Shoes | Resize-Convolution | Elimination artefacts |
| Instabilite | Tous | TTUR + Replay Buffer | Convergence robuste |

### Techniques transferables

Ces ameliorations ne sont pas specifiques a un dataset. Elles constituent une **boite a outils** applicable a tout probleme de traduction image-a-image :

1. **Spectral Norm + LSGAN** : Stabilisation universelle
2. **VGG Loss** : Qualite perceptuelle
3. **Resize-Convolution** : Elimination artefacts
4. **Self-Attention** : Dependances spatiales longue portee

---

## Transition vers Partie 3

Nous avons presente la theorie et les resultats quantitatifs. Passons maintenant a la demonstration concrete de nos modeles entraines, incluant une application surprise : la synesthesie artificielle.

---

# PARTIE 3 : Demonstration
## Duree cible : 2 minutes

---

## Slide 16 : Demonstration en direct - Inference
**Duree : 45 secondes**

Passons maintenant a la demonstration concrete de nos modeles.

### Procedure d'inference

Je vais executer une inference en temps reel sur nos trois modeles entraines :

1. **Maps** : Conversion d'un schema cartographique en vue satellite
2. **Facades** : Generation d'une facade a partir d'une segmentation
3. **Edges2Shoes** : Synthese d'une chaussure depuis un croquis

### Observations a noter

Pour chaque resultat, observez :
- La **coherence structurelle** : Les elements sont-ils positionnes correctement ?
- La **qualite des textures** : Les details sont-ils nets ou flous ?
- L'**absence d'artefacts** : Y a-t-il des motifs en damier ou des discontinuites ?

*[Execution des inferences sur les trois modeles]*

Comme vous pouvez le constater, les ameliorations implementees (VGG Loss, Resize-Convolution, Self-Attention) produisent des resultats significativement superieurs a la baseline.

---

## Slide 17 : Test sur croquis personnel
**Duree : 30 secondes**

### Objectif

Tester la capacite de generalisation du modele Edges2Shoes sur une entree hors distribution.

### Protocole

J'ai dessine un croquis de chaussure a la main, volontairement different des exemples d'entrainement :
- Lignes moins precises
- Style de dessin personnel
- Forme atypique

*[Affichage du croquis personnel et execution de l'inference]*

### Analyse du resultat

| Critere | Observation |
|---------|-------------|
| Reconnaissance forme | Le modele identifie correctement qu'il s'agit d'une chaussure |
| Generation texture | Textures coherentes malgre l'input inhabituel |
| Respect contours | Les grandes lignes sont preservees |
| Limites | Details fins parfois interpretes differemment |

**Conclusion :** Le modele generalise raisonnablement bien aux entrees hors distribution, demontrant un apprentissage de concepts semantiques plutot qu'une simple memorisation.

---

## Slide 18 : Synesthesie artificielle - Application experimentale
**Duree : 45 secondes**

### Concept

La synesthesie est un phenomene neurologique ou la stimulation d'un sens declenche automatiquement la perception d'un autre sens. Notre application tente de reproduire ce phenomene : **transformer un signal audio en representation visuelle**.

### Pipeline technique

```
Audio WAV -> Spectrogramme Mel -> Pix2Pix -> Image synthetique -> Video 3D cinetique
```

### Demonstration

*[Lecture d'un extrait audio avec affichage simultane de la video generee]*

Le modele a appris a associer :
- **Frequences basses** : Couleurs chaudes, formes larges
- **Frequences hautes** : Couleurs froides, details fins
- **Rythme** : Mouvement et pulsation dans la video

### Limites et perspectives

| Aspect | Etat actuel | Amelioration possible |
|--------|-------------|----------------------|
| Dataset | Synthetique | Annotations humaines de synesthetes reels |
| Diversite | Patterns predefinies | Apprentissage non supervise |
| Temps reel | Post-traitement | Inference optimisee GPU |

---

## Slide 19 : Conclusion et perspectives
**Duree : 30 secondes**

### Contributions principales

1. **Diagnostic systematique** des limites de la baseline Pix2Pix
2. **Validation experimentale** de techniques d'amelioration (VGG Loss, LSGAN, Self-Attention, Resize-Convolution)
3. **Application creative** : Synesthesie artificielle audio-visuelle

### Resultats quantitatifs

| Dataset | Amelioration principale | Impact |
|---------|-------------------------|--------|
| Maps | Self-Attention + LSGAN | Routes continues, textures realistes |
| Facades | Analyse des limites lambda | Identification du probleme structurel |
| Edges2Shoes | VGG Loss + Resize-Conv | G-loss record (7.96), zero artefact |

### Perspectives de recherche

- **Pix2PixHD** : Resolution superieure (1024x1024)
- **SPADE** : Normalisation spatiale adaptative
- **Diffusion Models** : Alternative aux GANs pour la generation d'images

### Mot de fin

Ce projet illustre l'importance de la demarche scientifique en deep learning : chaque probleme observe a conduit a une hypothese, un test, et une amelioration mesurable. Les techniques validees sont transferables a d'autres domaines de traduction image-a-image.

Merci de votre attention. Je suis disponible pour vos questions.

---

## Questions anticipees

### Q1 : Pourquoi ne pas avoir utilise les Diffusion Models ?

**Reponse :** Les Diffusion Models (DDPM, Stable Diffusion) produisent des resultats superieurs mais avec un cout computationnel significativement plus eleve. L'objectif pedagogique etait d'explorer les GANs conditionnels et leurs ameliorations. Cependant, une comparaison Pix2Pix vs Diffusion serait une extension interessante.

### Q2 : Comment avez-vous choisi les valeurs de lambda ?

**Reponse :** Lambda = 100 est la valeur recommandee dans le papier original. Nos experimentations ont montre que lambda = 10 combine a la VGG Loss produit de meilleurs resultats perceptuels. Le choix optimal depend du compromis fidelite/realisme souhaite pour chaque application.

### Q3 : Le modele peut-il generaliser a d'autres types d'images ?

**Reponse :** Pix2Pix requiert un entrainement supervise sur des paires alignees. Pour de nouveaux domaines, il faut collecter un dataset approprie. Des architectures comme CycleGAN permettent la traduction non appariee mais avec des contraintes differentes.

### Q4 : Quelle est la complexite computationnelle du Self-Attention ?

**Reponse :** O(n^2) ou n est le nombre de positions spatiales. C'est pourquoi nous l'avons place a la resolution 32x32 (1024 positions) plutot qu'a la resolution originale 256x256 (65536 positions). Le compromis qualite/cout est optimal a cette echelle.

---

# Annexes

## References bibliographiques
1. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. CVPR.
2. Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2019). Self-attention generative adversarial networks. ICML.
3. Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral normalization for generative adversarial networks. ICLR.

## Glossaire technique
| Terme | Definition |
|-------|------------|
| Skip-connection | Connexion directe entre couches non adjacentes |
| PatchGAN | Discriminateur evaluant des patches locaux |
| Instance Normalization | Normalisation par image individuelle |
| Mode collapse | Generateur produisant une diversite limitee |
| Vanishing gradient | Gradient trop faible pour mise a jour efficace |
