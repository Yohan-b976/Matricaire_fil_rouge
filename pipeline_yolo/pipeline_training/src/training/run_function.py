import json
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import os
import random
import numpy as np


def infer_dataset_type(folder_name, config):
    """
    Trouve la clé de dataset correspondant au nom du dossier.
    """
    for key in config.names.keys():
        if key in folder_name:
            return key
    raise ValueError(f"Aucun type de dataset reconnu dans : {folder_name}")


def generate_yolo_yaml(output_path, dataset_path, nc, class_names):
    """
    Génère un fichier YAML pour YOLO.
    
    Parameters:
        output_path (str or Path): chemin du fichier YAML à créer
        dataset_path (str or Path): chemin racine des images
        nc (int): nombre de classes
        class_names (list): liste des noms de classes
    """
    # structure du YAML
    data_dict = {
        "path": str(dataset_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": class_names
    }

    # écriture YAML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(data_dict, f, sort_keys=False)

    print(f"YAML généré : {output_path}")


def iteration_yolo(iterations, name_mod, dataset_path,yaml_path, config, path_outs, name_dataset):
    for it in range(iterations):
        print(f"=== Itération {it+1}/{iterations} ===")
        yolo_run(it, name_mod, dataset_path, yaml_path, config, path_outs, name_dataset)


def yolo_run(it, name_mod, dataset_path, yaml_path, config, path_outs, name_dataset):

    model = YOLO(name_mod + ".pt")
    dataset_path = Path(dataset_path).resolve()
    yaml_path = Path(yaml_path).resolve()
    print("YAML:", yaml_path)

    random_seed = random.randint(0, 1_000_000)

    # ======================
    # TRAIN
    # ======================
    model.train(
        data=str(yaml_path),
        imgsz=config.imgsz,
        epochs=config.epochs,
        batch=config.batch,
        patience=config.patience,
        workers=8,
        cache=True,
        device=0,
        project=str(dataset_path),
        name="runs",
        exist_ok=True,
        seed=random_seed
    )

    # ======================
    # BEST MODEL
    # ======================
    best_model_path = dataset_path / "runs" / "weights" / "best.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"best.pt introuvable : {best_model_path}")

    best_model = YOLO(best_model_path)

    # ======================
    # VALIDATION
    # ======================
    metrics = best_model.val(
        data=str(yaml_path),
        split="val",
        save_json=True,
        verbose=False
    )

    # ======================
    # SUMMARY
    # ======================
    summary_list = metrics.summary()  # liste de dicts par classe

    # convertir np.int64 / np.float64 en int / float pour JSON
    cleaned_summary = []
    for cls_dict in summary_list:
        cleaned_cls = {}
        for k, v in cls_dict.items():
            if isinstance(v, (np.integer, np.int64)):
                cleaned_cls[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                cleaned_cls[k] = float(v)
            else:
                cleaned_cls[k] = v
        cleaned_summary.append(cleaned_cls)

    # structurer JSON
    run_data = {
        "iteration": it,
        "model": name_mod,
        "dataset": name_dataset,
        "summary": cleaned_summary
    }

    # ======================
    # SORTIE JSON
    # ======================
    out_dir = Path(path_outs)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON par itération
    iter_json_path = out_dir / f"{name_mod}_{name_dataset}_iter_{it}.json"
    with open(iter_json_path, "w") as f:
        json.dump(run_data, f, indent=4)
    print(f"JSON itération sauvegardé : {iter_json_path}")
