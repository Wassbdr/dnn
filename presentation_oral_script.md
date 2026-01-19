# Script de Presentation Orale - Pix2Pix : Traduction d'Image a Image

**Duree totale : 15 minutes**

---

# PARTIE 1 : Introduction - Baseline, Infrastructure et Donnees

## Duree cible : 5 minutes

---

## Slide 1 : Titre et contexte
**Duree : 45 secondes**

*[Afficher le slide titre avec le nom du projet et les visuels des trois datasets]*

Bonjour a tous et merci d'etre presents aujourd'hui.

Je vais vous presenter notre travail sur **Pix2Pix**, une architecture de reseaux antagonistes generatifs conditionnels - ou cGAN - dediee a la **traduction d'image a image**.

Mais qu'est-ce que la traduction d'image a image, concretement ? C'est la capacite de transformer une image d'un domaine vers un autre tout en preservant sa structure. Par exemple : transformer un croquis en photo realiste, ou une carte schematique en vue satellite.

Dans ce projet, nous avons explore **trois domaines d'application** :
- Premierement, la **traduction cartographique** avec le dataset Maps - transformer des cartes Google Maps en vues satellites
- Deuxiemement, la **generation architecturale** avec Facades - creer des photos de batiments a partir de segmentations semantiques
- Et troisiemement, la **synthese de produits** avec Edges2Shoes - generer des images de chaussures a partir de simples contours

En bonus, nous avons developpe une application experimentale de **synesthesie artificielle**, qui transforme des signaux audio en representations visuelles. J'y reviendrai en fin de presentation.

Commencons par comprendre l'architecture de base sur laquelle repose tout notre travail.

---

## Slide 2 : Architecture Baseline - Le Generateur U-Net
**Duree : 1 minute 15 secondes**

*[Afficher le schema de l'architecture U-Net avec les skip-connections visibles]*

L'architecture Pix2Pix, introduite par Isola et ses collegues en 2017, repose sur deux composants qui jouent un jeu antagoniste. Commencons par le premier : le **Generateur**.

Notre generateur utilise une architecture **U-Net**, et c'est un choix fondamental. Pourquoi U-Net ? Laissez-moi vous expliquer.

Le U-Net est une architecture **encoder-decoder**. L'encodeur compresse progressivement l'image d'entree - il extrait des features de plus en plus abstraites tout en reduisant la resolution spatiale. Le decodeur fait l'inverse : il reconstruit progressivement une image a haute resolution.

Mais la vraie magie du U-Net, ce sont les **skip-connections** - ces connexions de saut que vous voyez sur le schema. Elles relient directement chaque couche de l'encodeur a son homologue symetrique dans le decodeur.

*[Pointer les skip-connections sur le schema]*

Pourquoi c'est crucial ? Sans ces connexions, le decodeur devrait reconstruire tous les details fins uniquement a partir de la representation compressee - le bottleneck. Avec les skip-connections, les informations de haute resolution sont transmises directement. C'est ce qui permet au generateur de produire des images nettes et detaillees.

Concretement, notre implementation comprend :
- **8 blocs de downsampling** dans l'encodeur - on passe de 256x256 a 1x1
- **8 blocs d'upsampling** dans le decodeur - on remonte a 256x256
- Et un **Dropout a 50%** dans les trois premiers blocs du decodeur

Ce Dropout est important : il introduit de la stochasticite pendant l'entrainement, ce qui aide a eviter le **mode collapse** - un phenomene ou le generateur produit toujours la meme sortie quelle que soit l'entree.

---

## Slide 3 : Architecture Baseline - Le Discriminateur PatchGAN
**Duree : 1 minute**

*[Afficher le schema du PatchGAN avec la grille de patches]*

Passons au second composant : le **Discriminateur**. Son role est de distinguer les images reelles des images generees, forcant ainsi le generateur a s'ameliorer.

Mais attention, nous n'utilisons pas un discriminateur classique. Nous utilisons un **PatchGAN**, et c'est une innovation cle de Pix2Pix.

Un discriminateur classique prendrait l'image entiere et produirait une seule prediction : "reelle" ou "fausse". Le probleme ? Il se concentre sur les structures globales et ignore souvent les details locaux.

Le PatchGAN adopte une approche radicalement differente : il evalue l'image **region par region**. Concretement, chaque position de sortie du discriminateur correspond a un **patch de 70x70 pixels** dans l'image d'entree.

*[Montrer visuellement comment les patches couvrent l'image]*

Le discriminateur produit donc une **grille de predictions** - pour une image 256x256, on obtient une matrice 30x30 ou chaque cellule dit "ce patch est reel" ou "ce patch est faux".

Quel est l'avantage ? Chaque region locale est scrutee independamment. Si le generateur produit une texture bizarre dans un coin de l'image, le PatchGAN le detectera. Cette approche force le generateur a produire des **textures haute frequence coherentes** sur l'ensemble de l'image, pas seulement une structure globale correcte.

C'est particulierement important pour nos taches : une route doit avoir une texture d'asphalte realiste, une facade doit avoir des briques credibles, une chaussure doit avoir un cuir convaincant - partout dans l'image.

---

## Slide 4 : Fonctions de perte - Le coeur de l'entrainement
**Duree : 1 minute**

*[Afficher la formule de la loss avec une decomposition visuelle]*

Maintenant que nous avons nos deux reseaux, comment les entrainons-nous ? Tout repose sur la **fonction de perte**, et Pix2Pix en combine deux.

La formule globale est :

**Loss_totale = Loss_adversariale + lambda × Loss_L1**

Decomposons ces deux termes.

**Premier terme : la perte adversariale**. C'est une Binary Cross-Entropy classique. Elle cree le jeu antagoniste : le generateur essaie de tromper le discriminateur, le discriminateur essaie de ne pas se faire tromper. Cette perte pousse le generateur a produire des images **visuellement realistes** - des textures credibles, des details convaincants.

**Deuxieme terme : la perte L1**. C'est simplement la difference absolue pixel par pixel entre l'image generee et l'image cible. Cette perte garantit la **fidelite structurelle** - l'image generee doit ressembler a la cible attendue, pas juste etre realiste.

*[Mettre en evidence le lambda = 100]*

Et ici, attention au parametre **lambda = 100**. Dans la baseline originale, la perte L1 est ponderee **cent fois** plus que la perte adversariale.

Pourquoi ce choix ? Les auteurs voulaient garantir que l'image generee corresponde a la cible. Sans cette contrainte forte, le generateur pourrait produire des images realistes mais completement differentes de ce qu'on attend.

Mais - et c'est un point critique que nous explorerons plus tard - ce lambda = 100 a un effet secondaire : il **privilegie la fidelite au detriment du realisme des textures**. Retenez ce point, il sera au coeur de nos optimisations.

---

## Slide 5 : Infrastructure technique
**Duree : 45 secondes**

*[Afficher un tableau recapitulatif de l'infrastructure]*

Avant de parler des donnees, un mot rapide sur notre **environnement technique**.

Nous avons travaille sur **Kaggle** et **Google Colab**, en utilisant des GPU **NVIDIA Tesla T4** avec 16 Go de VRAM. Le framework utilise est **PyTorch**.

Concernant la configuration d'entrainement, plusieurs choix meritent explication :

Le **batch size de 1** peut sembler etrange, mais il est impose par notre choix de normalisation. Nous utilisons **Instance Normalization** plutot que Batch Normalization. La difference ? Instance Norm calcule les statistiques pour chaque image individuellement, pas sur le batch entier. C'est plus adapte aux taches de style transfer et de generation d'images.

Pour le dataset Maps, une epoque prend environ **1 minute 30** - c'est raisonnable pour experimenter rapidement.

Cote sauvegarde, nous avons mis en place un systeme de **checkpointing** : le meilleur modele est sauvegarde automatiquement selon la validation loss. Et nous utilisons l'**early stopping** avec une patience de 10 epoques - si le modele ne s'ameliore pas pendant 10 epoques consecutives, on arrete l'entrainement pour eviter le surapprentissage.

---

## Slide 6 : Dataset - Source et composition
**Duree : 1 minute**

*[Afficher des exemples visuels de chaque sous-dataset]*

Parlons maintenant des donnees, car en deep learning, la qualite du dataset determine largement la qualite des resultats.

Nous utilisons le **Pix2Pix Dataset** disponible sur Kaggle, compile par Vikram Tiwari. C'est un dataset consequent : **2.7 Go**, environ **56 000 images** au total.

Le format est particulier et merite explication. Chaque fichier image est une **concatenation horizontale** de deux images : l'input a gauche, la cible a droite. Le fichier fait donc 512 pixels de large pour 256 de haut, et on le separe en deux images de 256x256.

*[Montrer visuellement la separation]*

Le dataset contient **quatre sous-ensembles**, chacun avec une tache specifique :

**Maps** : des paires extraites de Google Maps. A gauche, la carte schematique avec les routes en blanc, les parcs en vert. A droite, la vue satellite correspondante. La tache : apprendre a generer des vues satellites realistes.

**Facades** : des facades de batiments avec leur segmentation semantique. Chaque couleur represente un element - bleu pour les fenetres, rouge pour les portes, etc. La tache : transformer ces annotations colorees en photos de batiments credibles.

**Edges2Shoes** : des contours de chaussures - de simples lignes noires sur fond blanc - et les photos correspondantes. La tache : donner vie a ces croquis minimalistes.

**Cityscapes** : des scenes urbaines avec leur segmentation. Nous ne l'avons pas utilise dans ce projet, mais le principe est similaire.

Un point d'attention : **l'ordre input/target peut varier** selon le sous-dataset. J'ai perdu du temps au debut sur Facades parce que j'avais inverse les deux. Verification visuelle indispensable !

---

## Slide 7 : Dataset Synesthesie - Generation procedurale
**Duree : 30 secondes**

*[Afficher des exemples de paires audio-visuel]*

Pour notre application de synesthesie artificielle, nous avons du **creer notre propre dataset** - il n'en existe pas de pre-etabli pour cette tache.

Nous avons genere des paires de maniere **procedurale** : pour chaque type de signal audio, nous avons defini un pattern visuel correspondant.

Par exemple :
- Une **sinusoide pure** genere un **gradient vertical** doux
- Un **chirp** (frequence qui monte) produit des **ondulations**
- Une **modulation FM** cree des **diagrammes de Voronoi** complexes
- Des **percussions** generent des **formes radiales** explosives

L'avantage de cette approche : un controle total sur les correspondances. L'inconvenient : le modele pourrait ne pas generaliser aux sons reels, puisqu'il n'a vu que des patterns synthetiques. C'est une limite assumee de cette experimentation.

---

## Slide 8 : Problemes identifies dans la baseline
**Duree : 1 minute 15 secondes**

*[Afficher les courbes de loss montrant les problemes]*

Maintenant, la partie interessante. Apres avoir implemente la baseline et lance nos premiers entrainements, nous avons observe **trois problemes majeurs** qui ont guide toute la suite de notre travail.

**Premier probleme : l'effondrement du discriminateur.**

*[Montrer la courbe de D-loss qui s'effondre]*

Regardez cette courbe. A l'epoque 40, la D-loss est deja a 10 puissance -5. A l'epoque 60, elle atteint 1.7 fois 10 puissance -6. C'est minuscule.

Qu'est-ce que ca signifie ? Le discriminateur a **trop bien appris**. Il distingue parfaitement les vraies images des fausses, avec une confiance proche de 100%.

Et le probleme, c'est le **vanishing gradient**. Quand le discriminateur est aussi confiant, les gradients qu'il transmet au generateur deviennent quasi-nuls. Le generateur n'a plus de signal pour s'ameliorer. Il stagne.

**Deuxieme probleme : le flou excessif.**

*[Montrer des exemples d'images floues]*

Nos images generees manquaient systematiquement de details fins. Tout avait un aspect lisse, comme vu a travers un verre depoli.

L'explication est directement liee au lambda = 100 dont je parlais. Quand la perte L1 domine autant, quelle est la strategie optimale pour le generateur face a l'incertitude ? Generer la **moyenne des textures possibles**. Et une moyenne, c'est flou par definition.

**Troisieme probleme : le desequilibre D/G.**

*[Montrer les courbes G-loss vs D-loss]*

Le meilleur G-loss atteint etait de 27.4 a l'epoque 49. Ensuite, l'early stopping a arrete l'entrainement a l'epoque 129, mais la tendance etait claire : l'ecart entre D-loss et G-loss ne cessait de croitre.

Le systeme etait piege dans un **equilibre sous-optimal** : le discriminateur gagnait toujours, le generateur ne progressait plus.

---

## Slide 9 : Transition - Feuille de route des ameliorations
**Duree : 30 secondes**

*[Afficher un schema des solutions proposees]*

Ces trois problemes ont defini notre **feuille de route experimentale**. Pour chaque probleme, nous avons identifie une ou plusieurs solutions a tester.

Pour l'**effondrement du discriminateur**, nous allons explorer la **Spectral Normalization** - une technique qui contraint les poids du discriminateur pour eviter qu'il ne devienne trop puissant.

Pour le **flou excessif**, deux pistes : **reduire lambda** de 100 a 10, et ajouter une **VGG Perceptual Loss** qui evalue la similarite dans l'espace des features plutot que pixel par pixel.

Pour le **desequilibre D/G**, nous remplacerons la BCE par **LSGAN** - une formulation qui maintient des gradients utiles meme quand le discriminateur est confiant - et nous ajouterons un **learning rate scheduling** different pour D et G.

Dans la partie suivante, je vais vous montrer comment nous avons teste ces solutions sur nos trois datasets, en commencant par Facades, puis Maps, et enfin Edges2Shoes. Vous verrez que chaque dataset nous a appris quelque chose de different, et que les solutions se sont construites progressivement, d'une experimentation a l'autre.

Passons maintenant a l'analyse critique de nos experimentations.

---

# PARTIE 2 : Analyse critique des evolutions
## Duree cible : 8 minutes

---

## Slide 10 : Methodologie experimentale
**Duree : 45 secondes**

*[Afficher un schema du cycle experimental]*

Avant de plonger dans les resultats, permettez-moi de clarifier notre **demarche scientifique**. Car en deep learning, il est facile de tomber dans le piege du "j'ai change plein de trucs et ca marche mieux" - sans comprendre pourquoi.

Notre approche a ete rigoureuse. Pour chaque modification, nous avons suivi un cycle en cinq etapes :

**Premierement, l'hypothese.** On observe un probleme, on propose une explication theorique. Par exemple : "le flou vient du lambda trop eleve".

**Deuxiemement, l'implementation.** On modifie **un seul** parametre ou composant a la fois. C'est crucial - sinon, impossible de savoir ce qui a cause l'amelioration.

**Troisiemement, le test.** On lance un entrainement complet et on collecte les metriques.

**Quatriemement, l'analyse.** On compare avec la baseline. Est-ce que ca confirme ou infirme notre hypothese ?

**Et cinquiemement, l'iteration.** On ajuste et on recommence si necessaire.

Concernant les **metriques**, nous en avons utilise quatre principales :
- Le **FID** - Frechet Inception Distance - pour mesurer la qualite perceptuelle globale
- Le **LPIPS** pour la similarite perceptuelle entre images
- Et bien sur les **G-loss** et **D-loss** pour monitorer la stabilite de l'entrainement

Avec cette methodologie en tete, voyons maintenant nos experimentations, en commencant par Facades.

---

## Slide 11 : Facades - Confrontation aux limites de la baseline
**Duree : 1 minute 15 secondes**

*[Afficher des exemples de facades generees avec l'effet peinture]*

Facades a ete notre premier terrain d'experimentation, et honnement, c'est aussi celui ou nous avons le plus **appris de nos echecs**.

Le dataset Facades presente une tache particulierement exigeante. Regardez l'input : une segmentation semantique ou chaque couleur represente un element architectural - le bleu pour les fenetres, le rouge pour les portes, le gris pour les murs. Et la cible : une photo realiste de facade.

Le generateur doit donc accomplir deux choses simultanement : **respecter la disposition spatiale** encodee par les couleurs, et **synthetiser des textures realistes** - des briques, du verre, du bois.

*[Montrer les resultats avec l'effet peinture]*

Et voici ce que nous avons obtenu avec la baseline. Vous voyez le probleme ? Cet **effet peinture** caracteristique. Les facades ressemblent a des aquarelles, pas a des photos. Les textures sont lisses, les details fins sont absents.

Notre hypothese etait claire : c'est le **lambda = 100** qui cause ce flou. Rappelons la logique : quand la perte L1 domine, le generateur minimise la distance pixel-a-pixel. Face a l'incertitude - "quelle texture de brique exactement ?" - la strategie optimale est de generer la moyenne. Et une moyenne de textures, c'est flou.

Mais avant de modifier lambda, nous avons voulu verifier une chose : est-ce que plus d'entrainement resoudrait le probleme ?

*[Montrer le tableau comparatif]*

Nous avons entraine pendant 100 epoques, puis 200 epoques. Resultat ? L'effet peinture **persiste**. Ce n'est pas une question de convergence - le probleme est structurel.

**Conclusion importante :** Le probleme vient de la fonction de perte elle-meme, pas du temps d'entrainement. Cette observation a motive toute la suite de notre travail.

Mais plutot que de continuer sur Facades, nous avons decide de tester nos hypotheses sur **Maps** - un dataset qui partage le probleme du flou L1, mais qui ajoute un defi supplementaire : les **structures longue portee** comme les routes. Si nos solutions fonctionnent sur Maps, elles devraient etre transferables.

---

## Slide 12 : Maps - Les problemes specifiques
**Duree : 1 minute**

*[Afficher des exemples de cartes avec routes discontinues]*

Maps est un dataset fascinant pour tester Pix2Pix. L'input est une carte schematique Google Maps - routes blanches, parcs verts, batiments gris. La cible est la vue satellite correspondante.

Nous avons commence par la baseline pure : U-Net standard, PatchGAN, BCE plus L1 avec lambda = 100. Et nous avons observe **trois problemes distincts**.

*[Pointer les routes discontinues sur l'image]*

**Premier probleme : les routes discontinues.** Regardez cette intersection. La route arrive d'un cote... et ne ressort pas de l'autre. Le generateur n'arrive pas a maintenir la continuite des structures qui traversent l'image.

Pourquoi ? Pensez au PatchGAN. Il evalue des patches de 70x70 pixels. Une route peut faire 500 pixels de long. Le discriminateur ne "voit" jamais la route en entier - il ne peut donc pas penaliser les discontinuites.

**Deuxieme probleme : les textures plates.** Les zones residentielles, les parcs, les parkings - tout a la meme texture repetitive et artificielle. On retrouve l'effet du lambda trop eleve.

**Troisieme probleme : l'effondrement du discriminateur.** Notre vieille connaissance. La D-loss est tombee a 10 puissance -6. Le discriminateur distinguait parfaitement les vraies images des fausses, et le generateur ne recevait plus de gradient utile.

Ces trois problemes nous ont donne trois hypotheses a tester - et c'est la que les choses deviennent interessantes.

---

## Slide 13 : Maps - Self-Attention pour la coherence globale
**Duree : 1 minute 15 secondes**

*[Afficher le schema du mecanisme d'attention]*

Commencons par le probleme des routes discontinues. Notre hypothese : le generateur manque d'une vision **globale** de l'image.

Dans un U-Net classique, l'information se propage localement - chaque neurone ne "voit" qu'une petite region. Les skip-connections transmettent les details, mais pas les relations longue distance.

La solution ? Le **Self-Attention**.

*[Expliquer le schema]*

Le principe est elegant. Pour chaque position dans l'image, on calcule une "requete" - "qu'est-ce que je cherche ?". Pour toutes les autres positions, on calcule des "cles" - "qu'est-ce que j'offre ?". On compare les deux pour obtenir des poids d'attention, et on utilise ces poids pour agreger l'information.

En pratique, ca signifie que chaque pixel peut "regarder" tous les autres pixels de l'image et decider lesquels sont pertinents. Une route a gauche peut "voir" qu'il y a une route a droite et comprendre qu'elles doivent se connecter.

Ou placer cette couche d'attention ? C'est une question de **compromis cout-benefice**. L'attention a une complexite en O(n²) ou n est le nombre de positions. A la resolution originale 256x256, ca fait 65 000 positions - beaucoup trop couteux.

Nous l'avons placee apres le 4eme bloc de l'encodeur, a la resolution **32x32**. Ca fait environ 1000 positions - gerable. Et a cette echelle, chaque position represente deja une region de 8x8 pixels dans l'image originale, ce qui est suffisant pour capturer les structures globales.

*[Montrer le resultat avec routes continues]*

Le resultat ? Les routes se connectent maintenant correctement aux intersections. Le self-attention a resolu le probleme de coherence globale.

---

## Slide 14 : Maps - Stabiliser l'entrainement
**Duree : 1 minute**

*[Afficher les courbes de D-loss avant/apres Spectral Norm]*

Passons au probleme de l'effondrement du discriminateur. Rappelons la situation : D-loss qui chute vers zero, vanishing gradient, generateur qui stagne.

Notre solution : la **Spectral Normalization**.

Qu'est-ce que c'est ? C'est une technique de normalisation des poids qui **contraint la constante de Lipschitz** du reseau. En termes simples, ca empeche le discriminateur de faire des predictions trop extremes, trop confiantes.

*[Montrer la formule simplifiee]*

Concretement, apres chaque mise a jour des poids, on divise la matrice de poids par sa plus grande valeur singuliere. Ca "calme" le discriminateur - il ne peut plus devenir infiniment confiant.

*[Comparer les courbes]*

Regardez la difference. A gauche, sans Spectral Norm : la D-loss s'effondre vers 10^-6. A droite, avec Spectral Norm : la D-loss reste dans une plage saine, entre 0.1 et 1.0.

Et l'effet sur le generateur ? Il continue a recevoir des gradients utiles tout au long de l'entrainement. Plus de stagnation.

Mais la stabilisation du discriminateur ne suffit pas. Nous avons aussi change la **fonction de perte adversariale** - et c'est notre prochaine amelioration.

---

## Slide 15 : Maps - LSGAN et VGG Loss
**Duree : 1 minute 15 secondes**

*[Afficher la comparaison BCE vs MSE]*

La Binary Cross-Entropy - notre perte adversariale de base - a un defaut fondamental. Quand le discriminateur est confiant, le gradient devient tres faible. C'est mathematiquement inevitable avec la forme -log(x).

La solution ? Remplacer BCE par **LSGAN** - Least Squares GAN. Au lieu de -log(D(x)), on utilise (D(x) - 1)².

*[Montrer les courbes de gradient]*

La difference est cruciale. Avec LSGAN, le gradient reste significatif meme quand le discriminateur est confiant. Un faux echantillon loin de la distribution reelle recoit une **penalite proportionnelle** a sa distance, pas juste un "c'est faux".

Resultat pratique : le generateur continue a s'ameliorer meme tard dans l'entrainement.

*[Transition vers VGG Loss]*

Maintenant, parlons de la **VGG Perceptual Loss** - l'amelioration qui a eu le plus d'impact sur la qualite visuelle.

L'idee est simple mais puissante. Au lieu de comparer les images pixel par pixel avec L1, on les compare dans l'**espace des features** d'un reseau VGG16 pre-entraine.

*[Montrer le schema VGG]*

On passe l'image generee et l'image cible dans VGG16, on extrait les activations a une couche intermediaire - relu3_3 dans notre cas - et on calcule la distance L1 entre ces activations.

Pourquoi c'est mieux ? VGG a appris a reconnaitre des textures, des formes, des patterns. Deux images avec des textures visuellement similaires auront des activations VGG proches, meme si les pixels individuels different. Le generateur apprend donc a produire des **textures perceptuellement coherentes**, pas des copies pixel-parfaites.

Et nous avons **reduit lambda de 100 a 10**. La VGG Loss compense la perte de contrainte L1 tout en privilegiant la qualite perceptuelle.

---

## Slide 16 : Transition Maps vers Edges2Shoes
**Duree : 30 secondes**

*[Afficher la configuration finale Maps]*

Recapitulons notre configuration finale pour Maps :
- L1 lambda reduit a 10
- VGG Loss avec lambda 10
- LSGAN au lieu de BCE
- Instance Normalization dans le generateur
- Spectral Normalization dans le discriminateur
- Self-Attention pour la coherence globale

Ces ameliorations constituent notre **nouvelle baseline** - validee, testee, fonctionnelle.

Mais Edges2Shoes pose un defi different. On ne genere plus des vues satellites avec des structures geometriques - on genere des **textures riches et variees** : du cuir, de la toile, des lacets, des coutures. Et l'input est minimal : de simples contours noirs sur fond blanc.

La question qui s'est posee : l'architecture U-Net, avec ses skip-connections qui forcent une correspondance spatiale stricte, est-elle vraiment optimale pour une tache aussi **creative** ?

J'ai voulu tester une alternative.

---

## Slide 17 : Edges2Shoes - L'experimentation ResNet
**Duree : 1 minute 30 secondes**

*[Afficher l'architecture ResNet vs U-Net]*

Mon hypothese etait la suivante : les skip-connections du U-Net sont peut-etre **trop contraignantes** pour Edges2Shoes.

Pensez-y. Les skip-connections transmettent directement les features de l'encodeur au decodeur. Pour Maps, c'est parfait - on veut que chaque route dans l'input corresponde exactement a une route dans l'output.

Mais pour Edges2Shoes ? L'input est un contour minimaliste. La majorite de l'information doit etre **inventee** par le generateur. Peut-etre qu'une architecture avec plus de liberte creative produirait de meilleures textures ?

J'ai donc teste un **generateur ResNet** - 9 blocs residuels, pas de skip-connections.

*[Montrer les resultats ResNet]*

Et les premiers resultats etaient... **encourageants**. Vraiment encourageants.

Regardez ces textures. Le cuir brille, la toile a du grain, les lacets ont des details. Les couleurs sont naturelles et variees. Le ResNet produisait des chaussures visuellement tres convaincantes.

*[Pause]*

Mais il y avait un probleme. Et il m'a fallu du temps pour le voir clairement.

*[Montrer les zones avec brouillard de pixels]*

Ce **brouillard de pixels**. Vous le voyez ? Ces halos grisatres autour des contours, cette perte de nettete dans les zones de transition. Ce n'etait pas systematique - certaines images etaient parfaites - mais quand le probleme apparaissait, il etait redhibitoire.

Le diagnostic ? Sans les skip-connections, le decodeur doit reconstruire **toute** l'information spatiale a partir du bottleneck compresse. Et parfois, il n'y arrive pas parfaitement. L'information se perd, et ca cree ce brouillard.

**Decision difficile** : malgre la richesse des textures, j'ai decide de revenir au U-Net. Le brouillard de pixels degradait trop la qualite finale.

Mais - et c'est important - cette experimentation n'etait pas un echec. Elle m'a montre que des textures riches etaient **possibles**. Le probleme etait la methode, pas l'objectif. Il fallait trouver un autre moyen d'enrichir les textures tout en gardant la precision spatiale du U-Net.

Et ce moyen, c'etait la VGG Loss.

---

## Slide 18 : Edges2Shoes - La revolution VGG Loss
**Duree : 1 minute**

*[Afficher les courbes de convergence spectaculaires]*

Plutot que de changer l'architecture, j'ai change la **fonction de perte**. Meme configuration que pour Maps : U-Net avec skip-connections, mais avec VGG Loss et lambda reduit a 10.

*[Montrer les metriques]*

Et la... regardez ces chiffres.

Avec la baseline L1 seule, le G-loss a l'epoque 2 etait autour de 45. Avec VGG Loss ? **7.96**. A l'epoque 2 !

C'est un record. Le reseau converge en moins de 10 epoques vers une qualite qui prenait 50+ epoques avant.

*[Montrer des exemples visuels]*

Et visuellement ? Les textures sont nettes, riches, detaillees. Le cuir a du brillant, la toile a du grain. On obtient la qualite des textures du ResNet **sans** le brouillard de pixels.

Pourquoi ca marche aussi bien ? La VGG Loss guide le generateur vers des **textures perceptuellement coherentes** des les premieres iterations. Le reseau apprend d'abord les caracteristiques haut niveau - la structure globale, le type de texture - avant de raffiner les details.

Avec L1 seule, le reseau passait des dizaines d'epoques a minimiser des differences pixel-a-pixel souvent non pertinentes. Avec VGG, il se concentre sur ce qui compte visuellement.

C'est exactement ce que je cherchais : la richesse du ResNet avec la precision du U-Net.

---

## Slide 19 : Edges2Shoes - Elimination des artefacts checkerboard
**Duree : 1 minute**

*[Montrer des exemples d'artefacts en damier]*

Mais il restait un dernier probleme. Malgre toutes ces ameliorations, certaines images presentaient des **motifs en damier** - des artefacts periodiques, particulierement visibles dans les zones unies.

*[Zoomer sur les artefacts]*

Vous les voyez ? Cette grille subtile qui apparait dans le fond, parfois sur la chaussure elle-meme.

Le coupable ? La **deconvolution** - ConvTranspose2d.

*[Expliquer avec un schema]*

Quand on fait de l'upsampling avec ConvTranspose2d, le kernel "ecrit" dans l'image de sortie avec un certain stride. Si le kernel_size n'est pas divisible par le stride, certains pixels de sortie recoivent plus de contributions que d'autres. Ca cree un motif periodique - le fameux damier.

Notre configuration utilisait kernel_size=4 et stride=2. 4 divise 2... mais le probleme persiste a cause des effets de bord.

*[Montrer la solution]*

La solution s'appelle **Resize-Convolution**. Au lieu de faire upsampling et convolution en une seule operation, on les separe :

D'abord, un **Upsample** avec interpolation bilineaire - ca double la resolution de maniere lisse, sans artefacts.

Ensuite, une **convolution standard** - qui apprend les features sans creer de motifs periodiques.

*[Montrer le resultat]*

Le resultat ? Zero artefact. Les zones unies sont parfaitement lisses, les textures sont preservees.

C'est un changement simple mais crucial. Et c'est typique de ce projet : des details d'implementation qui ont un impact enorme sur la qualite finale.

---

## Slide 20 : Configuration finale et synthese
**Duree : 1 minute**

*[Afficher le tableau recapitulatif complet]*

Recapitulons le chemin parcouru avec un tableau de synthese.

| Probleme | Ou observe | Solution | Impact |
|----------|------------|----------|--------|
| Effet peinture / flou | Facades, tous | VGG Loss + lambda=10 | Textures nettes |
| Effondrement discriminateur | Maps | Spectral Normalization | Entrainement stable |
| Vanishing gradient | Maps, Edges2Shoes | LSGAN (MSE) | Gradients maintenus |
| Routes discontinues | Maps | Self-Attention | Coherence globale |
| Brouillard de pixels | Edges2Shoes (ResNet) | Retour U-Net | Precision spatiale |
| Artefacts damier | Edges2Shoes | Resize-Convolution | Zero artefact |

*[Mettre en evidence la configuration finale]*

Notre configuration finale pour Edges2Shoes - la version 5.1 que nous appelons "Ultimate" - integre toutes ces ameliorations :

- Generateur U-Net avec Resize-Convolution
- Discriminateur PatchGAN avec Spectral Normalization
- Pertes : MSE (LSGAN) + L1 (lambda=10) + VGG (lambda=10)
- Stabilisation supplementaire : Replay Buffer et TTUR
- Batch size augmente a 4 pour plus de stabilite
- Early stopping avec patience de 10 epoques

Ce qui est important a retenir, c'est que ces techniques ne sont **pas specifiques** a un dataset. Elles constituent une **boite a outils** applicable a tout probleme de traduction image-a-image.

Vous travaillez sur un nouveau projet de generation d'images ? Commencez par cette configuration. Ajoutez Self-Attention si vous avez des structures longue portee. Ajustez les lambdas selon vos besoins de fidelite vs realisme.

Passons maintenant a la demonstration concrete de ces modeles en action.

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
