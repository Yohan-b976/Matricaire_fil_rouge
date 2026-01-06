# Pipeline YOLOv8 – Détection de la Matricaire

## Contexte du projet

Ce dépôt contient un **pipeline complet de détection d’objets** basé sur **YOLOv8**, appliqué à la détection de la **matricaire** (*Tripleurospermum inodorum*) à partir d’images.

Le projet est développé dans le cadre du **fil rouge IODAA (3A)** et vise à structurer de manière modulaire :
- la préparation des images,
- l’annotation,
- l’entraînement du modèle,
- les tests préliminaires.

---

## Objectifs

- Préparer et découper des images pour la détection
- Entraîner un modèle YOLOv8 sur un jeu de données annoté
- Tester rapidement le bon fonctionnement du pipeline
- Disposer d’un code clair, modulaire et reproductible

---

## Arborescence du projet

```text
pipeline_yolo/
├── annotation_sm_images/
│   └── readme.md
│
├── cut_images/
│   ├── images/
│   │   └── readme.md
│   └── cut_image.py
│
├── train_yolov8/
│   ├── dataset/
│   │   └── readme.me
│   ├── dataset_init.py
│   └── yolo_training.py
│
├── test_preliminaire/
│   └── test_yolov8/
│       └── test_yolo.py
│
├── .gitignore
└── README.md

