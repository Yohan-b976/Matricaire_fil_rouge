import os
import shutil
from preprocessing.boite_sol import add_background_boxes
from preprocessing.dataset_split import split_dataset
from pathlib import Path

def set_databases(
    soil_list, source_path, destination_path,
    database_name, config, seed=None
):
    """
    Crée des variantes de la base de données avec des boîtes de fond.
    Le background_class_id est déterminé depuis config.names en recherchant un match partiel.
    """
    if not os.path.exists(source_path):
        print(f"⚠️ Source introuvable : {source_path}")
        return

    # =========================
    # Création du dossier intermédiaire
    # =========================
    intermed_path = os.path.join(destination_path, "intermed")
    os.makedirs(intermed_path, exist_ok=True)
    print(f"✔ Dossier intermédiaire prêt → {intermed_path}")

    # =========================
    # Copie de la base originale dans intermed
    # =========================
    original_dst = os.path.join(intermed_path, f"{database_name}")
    shutil.copytree(source_path, original_dst, dirs_exist_ok=True)
    print(f"✔ Copie originale créée → {original_dst}")

    # =========================
    # Ajout des variantes avec boîtes de fond
    # =========================
    for nb_boxes in soil_list:
        variant_name = f"{database_name}_bg{nb_boxes}"
        dst = os.path.join(intermed_path, variant_name)

        shutil.copytree(source_path, dst, dirs_exist_ok=True)
        print(f"✔ Copié → {variant_name}")

        # =========================
        # Détermination automatique de l'ID de fond
        # =========================
        matched_key = None
        for key in config.id_backgroung.keys():
            if key.lower() in database_name.lower() or database_name.lower() in key.lower():
                matched_key = key
                print(f"✔ Match trouvé pour '{database_name}' → '{matched_key}' (ID de fond: {config.id_backgroung[matched_key]})")
                break

        background_class_id = config.id_backgroung[matched_key]

        add_background_boxes(
            dst,
            background_class_id=background_class_id,
            nb_boxes=nb_boxes,
            mu=config.mu,
            sigma=config.sigma,
            max_tries=config.max_tries,
            seed=seed
        )

        print(f"✔ Background ajouté ({nb_boxes} boîtes/image) avec ID {background_class_id}")





# def set_databases(
#     soil_list, source_path, destination_path,
#     database_name, mu=18,
#     sigma=10, max_tries=1000, background_class_id = 2, 
#     seed=None
# ):

#     if not os.path.exists(source_path):
#         print(f"⚠️ Source introuvable : {source_path}")
#         return

#     # =========================
#     # Création du dossier intermédiaire
#     # =========================
#     intermed_path = os.path.join(destination_path, "intermed")
#     os.makedirs(intermed_path, exist_ok=True)  # crée le dossier s'il n'existe pas
#     print(f"✔ Dossier intermédiaire prêt → {intermed_path}")

#     # =========================
#     # Copie de la base originale dans intermed
#     # =========================
#     original_dst = os.path.join(intermed_path, f"{database_name}")
#     shutil.copytree(source_path, original_dst, dirs_exist_ok=True)
#     print(f"✔ Copie originale créée → {original_dst}")

#     # =========================
#     # Ajout des variantes avec boîtes de fond
#     # =========================
#     for nb_boxes in soil_list:
#         variant_name = f"{database_name}_bg{nb_boxes}"
#         dst = os.path.join(intermed_path, variant_name)

#         shutil.copytree(source_path, dst, dirs_exist_ok=True)
#         print(f"✔ Copié → {variant_name}")

#         add_background_boxes(
#             dst,
#             background_class_id=background_class_id,
#             nb_boxes=nb_boxes,
#             mu=mu,
#             sigma=sigma,
#             max_tries=max_tries,
#             seed=seed
#         )

#         print(f"✔ Background ajouté ({nb_boxes} boîtes/image)")


def dataset_preparation(source_path, database_name, dataset_number):
    """
    Prépare un dataset YOLO en créant un dossier
    dataset_<database_name> puis en splittant les données dedans.
    """

    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source introuvable: {source_path}")

    for i in range(dataset_number):
            # nom du dossier final pour ce split
            dataset_dir = source_path.parent / f"dataset_{database_name}_split{i}"

            # création du dossier parent
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # appel du split
            split_dataset(src=source_path, out=dataset_dir, seed=42+i)

            print(f"\n✔ Dataset split {i} prêt : {dataset_dir}")
