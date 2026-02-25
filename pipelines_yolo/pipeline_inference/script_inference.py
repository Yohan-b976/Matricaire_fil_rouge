# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
from shapely.geometry import Polygon
from shapely import wkt
import time

# =========================================================
# Vérification et padding
# =========================================================
def ensure_multiple_of_126(img):
    H, W = img.shape[:2]

    pad_h = (126 - H % 126) % 126
    pad_w = (126 - W % 126) % 126

    if pad_h != 0 or pad_w != 0:
        """print(
            f"[INFO] Image non multiple de 126. Padding appliqué : "
            f"+{pad_h} pixels en bas, +{pad_w} pixels à droite"
        )"""

        padded = np.zeros((H + pad_h, W + pad_w, 3), dtype=np.uint8)
        padded[:H, :W] = img  # l'image originale reste en haut à gauche
        return padded, pad_h, pad_w

    return img, 0, 0

# =========================================================
# Fonctions pour la cartographie des fleurs
# =========================================================
def get_drone_image_Lambert_93_coord(coord_folder):
    """Retrieve Lambert 93 coords from folder.

    Args:
        coord_folder (_type_): folder containing all the Lambert 93 coords of the drone images.
        
    Returns:
        coords (dict): dict with a tuple containing the lambert 93 coords for each drone image.
        NB. Lambert 93 format (origin is upper left corner of image)
        (pixel resolution in x, rotation angle 1, rotation angle 2, 
        resolution in y, Lambert 93 origin x, Lambert 93 origin y)
    """
    coords = dict()
    for filename in os.listdir(coord_folder):
        if filename.endswith('.JGW'):
            drone_image_name = "_".join(filename.split("_")[0:-1]) 
            with open(os.path.join(coord_folder, filename), "r") as f:
                coord_list = list()
                line_count = 0
                for line in f:
                    line_count += 1
                    values = line.split()                    
                    floats = [float(value) for value in values]
                    coord_list.append(floats[0])
                if line_count != 6:
                    raise ValueError(
                        f"Invalid Lambert 93 coordinates: file {filename}, "
                        f"{line_count} lines found (expected 6)"
                    )                    
            coords[drone_image_name] = tuple(coord_list)
    #apply rotation to change the origin of image
    return coords    

def convert_in_Lambert_93(coord_drone: tuple, drone_bb: list, img_width: int) -> tuple:
    """
    Convert YOLO bounding boxes (from rotated/padded landscape image) into Lambert 93 coordinates
    for the original portrait-oriented drone image.

    Args:
        coord_drone (tuple): Lambert 93 coordinates of the drone image
            (res_X, rot_angle1, rot_angle2, res_Y, origin_X, origin_Y)
            rot_angle1 and rot_angle2 must be 0 (no rotation in Lambert file)
        drone_bb (list): list of YOLO bounding boxes [(x, y, w, h), ...] in rotated/padded image
        img_height (int): height of original image (before rotation/padding)
        img_width (int): width of original image (before rotation/padding)


    Returns:
        tuple of tuples: Lambert 93 coordinates of bounding boxes
                         ((X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4)), ...
    """
    # Check rotation angles
    if coord_drone[1] != 0 or coord_drone[2] != 0:
        raise ValueError("Rotation angles in Lambert 93 coords not supported")

    res_X, _, _, res_Y, origin_X, origin_Y = coord_drone
    L93_drone_bb_list = []

    for bb in drone_bb:
        # bb = (x, y, w, h) in rotated landscape image
        x_rot, y_rot, w_rot, h_rot = bb

        # Map back to portrait coordinates (undo 90° CCW rotation)
        x_portrait = img_width - (y_rot + h_rot)
        y_portrait = x_rot
        w_portrait = h_rot
        h_portrait = w_rot

        # Convert pixels to Lambert 93 coordinates
        X = origin_X + x_portrait * res_X
        Y = origin_Y + y_portrait * res_Y
        W = w_portrait * res_X
        H = h_portrait * res_Y

        # Polygon: top-left, top-right, bottom-right, bottom-left
        L93_drone_bb_list.append(
            ((X, Y), (X + W, Y), (X + W, Y + H), (X, Y + H))
        )

    return tuple(L93_drone_bb_list)

def wheat_and_peas_plots_coords(plot_csv):
    plots = pd.read_csv(plot_csv)
    plots_coords = {}
    for index in range(plots.shape[0]):
        plot_num = plots['XY'].iloc[index]
        test = plots['WKT'].iloc[index]
        geom = wkt.loads(test)
        minx, miny, maxx, maxy = geom.bounds
        plots_coords[plot_num] = (minx, miny, maxx, maxy)
    return plots_coords

def flower_in_plot(plot_coords, flower_bounding_box):
    center_bb_x = flower_bounding_box[0][0] + (flower_bounding_box[1][0] - flower_bounding_box[0][0]) / 2
    center_bb_y = flower_bounding_box[2][1] + (flower_bounding_box[1][1] - flower_bounding_box[2][1]) / 2
    min_x, min_y, max_x, max_y = plot_coords
    return min_x <= center_bb_x <= max_x and min_y <= center_bb_y <= max_y
    
def qgis_wkt_csv_file(output_folder, bb_L93_coords):
    """
    Creates a csv file with the coordinates of the bounding boxes in polygons in WKT format in the Lambert 93 for QGIS.
    """
    WKT = list()
    for bb in bb_L93_coords:
        WKT.append(Polygon(bb).wkt)
    n = len(bb_L93_coords)
    d = {"number": [k for k in range(n)], "WKT": WKT}
    df = pd.DataFrame(data=d)
    df.to_csv(output_folder+'/coordonnees_QGIS_fleurs'+'.csv', index=False) 


# =========================================================
# Pipeline YOLO encapsulée
# =========================================================
class YoloTilingPipeline:
    def __init__(self, model_path, output_folder):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)  
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Paramètres
        self.TILE_SIZE = 126
        self.OVERLAP = 0
        self.STRIDE = int(self.TILE_SIZE * (1 - self.OVERLAP))
        self.MARGIN_SIZE = 20
        self.CONF_THRES = 0.25

    # =====================================================
    # Inférence sur une seule image
    # =====================================================
    def predict_image(self, image_path, visualize):
        start = time.time()

        filename = os.path.basename(image_path)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARNING] Impossible de lire {image_path}")
            return None, None, None, None, None  # in case of failure
    
        # Original size
        orig_H, orig_W = img.shape[:2]
    
        # Rotate 90° CCW to landscape for YOLO
        img = np.rot90(img)
    
        # Ensure multiple of 126
        img, pad_h, pad_w = ensure_multiple_of_126(img)
        H, W = img.shape[:2]
    
        offset = self.TILE_SIZE // 2
        black_image = np.zeros((H + self.TILE_SIZE, W + self.TILE_SIZE, 3), dtype=np.uint8)
        black_image[offset:H + offset, offset:W + offset] = img
        img = black_image
    
        traitement_img = img.copy()
        output_img = img.copy()
    
        all_boxes = []
        total_boxes = 0
    
        # Sliding window coordinates
        coords = [
            (offset, H - 3 * offset, offset, W - 3 * offset),
            (offset, H - 3 * offset, 0, W - offset),
            (0, H - offset, offset, W - 3 * offset),
            (0, H - offset, 0, W - offset),
        ]
    
        for coord in coords:
            for y in range(coord[0], coord[1], self.STRIDE):
                for x in range(coord[2], coord[3], self.STRIDE):
                    tile = traitement_img[y:y + self.TILE_SIZE, x:x + self.TILE_SIZE]
    
                    results = self.model(
                        tile,
                        imgsz=128,
                        conf=self.CONF_THRES,
                        verbose=False,
                        classes=[0],
                        device=self.device,
                        half=True
                    )
    
                    x1_safe = y1_safe = self.MARGIN_SIZE
                    x2_safe = y2_safe = self.TILE_SIZE - self.MARGIN_SIZE
    
                    for r in results:
                        if r.boxes is None:
                            continue
                        boxes = r.boxes.xyxy.detach().cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            if x2 < x1_safe or y2 < y1_safe or x1 > x2_safe or y1 > y2_safe:
                                continue
    
                            x1_full = int(x1 + x)
                            y1_full = int(y1 + y)
                            x2_full = int(x2 + x)
                            y2_full = int(y2 + y)
    
                            all_boxes.append(
                                (x1_full - offset, y1_full - offset, x2_full - x1_full, y2_full - y1_full)
                            )
    
                            # Mask overlapping area
                            cv2.rectangle(traitement_img, (x1_full, y1_full), (x2_full, y2_full), (0, 0, 0), -1)
                            # Draw detection
                            if visualize:
                                cv2.rectangle(output_img, (x1_full, y1_full), (x2_full, y2_full), (0, 0, 255), 1)
                            total_boxes += 1
    
                    # Mask safe area
                    cv2.rectangle(traitement_img, (x + x1_safe, y + y1_safe), (x + x2_safe, y + y2_safe), (0, 0, 0), -1)
    
        # Save output image
        if visualize :
            out_path = os.path.join(self.output_folder, filename)
            cv2.imwrite(
                out_path,
                output_img[offset:H - pad_h + offset, offset:W - pad_w + offset]
            )
        print(f"[INFO] Temps de traitement par YOLO: {time.time() - start}s")
        print(f"[INFO] Détection de {total_boxes} fleurs dans {filename}")
    
        # Return all data needed for Lambert conversion
        return all_boxes, orig_W
    # =====================================================
    # Batch sur dossier
    # =====================================================
    def predict_folder(self, input_folder, output_folder, plots_coords_path, visualize):
        Lambert_93_images_coord = get_drone_image_Lambert_93_coord(input_folder)
        Lambert_93_plots_coord = wheat_and_peas_plots_coords(plots_coords_path)
        bounding_boxes_L93_coords = list()
        plots_count_summary = dict()
        os.makedirs(output_folder, exist_ok=True)
    
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".jpg"):
                img_path = os.path.join(input_folder, filename)
                parts = filename.split("_")
                drone_image_name = "_".join(parts[:-1])
                plot = "_".join(parts[:2])
    
                print("[INFO] Ouverture de l'image:", filename)
    
                if drone_image_name not in Lambert_93_images_coord:
                    print(f"[WARNING] Coordonnées absentes pour {drone_image_name}")
                    continue
    
                if plot not in Lambert_93_plots_coord:
                    print(f"[WARNING] Parcelle inconnue: {plot}")
                    continue
                print(f"[INFO] YOLO traite l'image...")
                file_coords, orig_W = self.predict_image(img_path, visualize)
    
                if file_coords is None:
                    print(f"[INFO] {filename} ne contient pas de fleurs.")
                    continue
                
                file_L93_bb = convert_in_Lambert_93(
                    Lambert_93_images_coord[drone_image_name],
                    file_coords,
                    img_width=orig_W,
                )
                print(f"[INFO] Focus sur la parcelle {plot} au centre de l'image")
                count_flower_plot = 0
                for bb in file_L93_bb:
                    if flower_in_plot(Lambert_93_plots_coord[plot], bb):
                        bounding_boxes_L93_coords.append(bb)
                        count_flower_plot += 1
                plots_count_summary[plot] =  count_flower_plot
                print(f"[INFO] Détection de {count_flower_plot} fleurs dans la parcelle {plot}")
    
        print("[INFO] Création du fichier CSV WKT pour QGIS")
        qgis_wkt_csv_file(output_folder, bounding_boxes_L93_coords)
        print("[INFO] Création du fichier CSV avec les comptes de fleurs par parcelle")
        count_summary = pd.DataFrame({
            "parcelle": plots_count_summary.keys(),
            "nombre de matricaires": plots_count_summary.values()
        })
        count_summary.to_csv(output_folder+'/comptage'+'.csv', index=True) 
# =========================================================
# Exemple d'utilisation
# =========================================================
if __name__ == "__main__":

    model_path = "data/yolo26s_bg1_activelearning/best.pt"
    output_folder = "./resultats"
    input_folder = "data/photos"
    plots_coords_path = 'data/grille_parcelles.csv'
    visualize = True

    pipeline = YoloTilingPipeline(model_path, output_folder)
    pipeline.predict_folder(input_folder, output_folder, plots_coords_path, visualize)