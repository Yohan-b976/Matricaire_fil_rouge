# Pipeline YOLOv8 – Détection de la Matricaire

## Contexte du projet

Ce dépôt contient un **pipeline complet de détection d’objets** basé sur **YOLOv8**, appliqué à la détection de la **matricaire** (*Tripleurospermum inodorum*) à partir d’images.

Le projet est développé dans le cadre du **fil rouge IODAA (3A)** et vise à structurer de manière modulaire :
- la préparation des images,
- l’annotation,
- l’entraînement du modèle,
- les tests préliminaires.

---

# Organisation du projet

Ce dépôt contient l’ensemble des outils permettant de préparer les données, entraîner des modèles YOLO et analyser les performances.

---

## Arborescence

.
├── archives/
├── cut_images/
│   └── cut_image.py
├── inference/
│   └── YOLO_running_focus.py
├── pipelines_yolo/pipeline_training/
│   ├── data/raw/
│   │   └── readme.md
│   └── src/
│       ├── preprocessing/
│       │   ├── boite_sol.py
│       │   └── dataset_split.py
│       ├── preprocessing_pipeline/
│       │   └── database_preparation.py
│       ├── stats/
│       │   └── stats_flower.py
│       ├── training/
│       │   └── run_function.py
│       ├── config.py
│       └── main.py
├── ultralytics_a_remplacer/
│   ├── metrics.py
│   ├── plotting.py
│   └── val.py
├── .gitignore
└── README.md


---

## archives/

Contient l’ancien travail réalisé lors des expérimentations d’entraînement de plusieurs modèles.

Ce dossier sert de référence historique et de sauvegarde des tests précédents.

---

## cut_images/

Outils de découpage d’images.

**cut_image.py**  
Segmente les images en tuiles (tiles) afin de permettre leur traitement par YOLO lorsque les images originales sont trop grandes.

---

## inference/

Scripts dédiés à l’inférence.

**YOLO_running_focus.py**  
Permet de tester une méthode de yolo avec shift d'attention sur les images permettant d'éviter les erreurs de découpages et de double comptage de fleur.

---

## pipelines_yolo/

Contient les pipelines du projet.

> ⚠️ Pour le moment, un seul pipeline est implémenté.

### pipeline_training/

Pipeline complet pour l’entraînement des modèles.

Ce pipeline permet d’entraîner des modèles YOLO à partir d’une base d’images annotées avec **LabelImg**.

#### data/raw/

Contient les données brutes.

#### src/

Regroupe tous les modules utilisés dans le pipeline.

##### preprocessing/

Prétraitement des données :

- génération des boîtes sol  
- découpage des datasets  
- préparation des annotations  

##### preprocessing_pipeline/

**database_preparation.py**  
Préparation et structuration de la base d’entraînement.

##### stats/

**stats_flower.py**  
Calcul des statistiques sur les données et les résultats.

##### training/

**run_function.py**  
Fonctions de lancement des entraînements.

##### config.py

Fichier de configuration du pipeline.

##### main.py

Point d’entrée principal pour exécuter le pipeline.

---

## ultralytics_a_remplacer/

Contient les fichiers modifiés d’**Ultralytics YOLO**.

Ces fichiers doivent remplacer ceux du package Ultralytics afin d’intégrer des métriques personnalisées basées sur des **ratios**.

- `metrics.py` → calcul des métriques personnalisées  
- `plotting.py` → visualisation adaptée  
- `val.py` → validation intégrant les nouvelles métriques  

---

## Objectif du projet

Ce projet fournit une pipeline complète permettant :

- la préparation et le découpage des images  
- l’entraînement de modèles YOLO  
- l’évaluation avec métriques personnalisées  
- l’analyse statistique des performances  









