class Config:
    def __init__(self):
        self.modeles = ["yolov8"]#, "yolo11", "yolo26"
        self.size = [ "l"]#"n", "s", "m",
        self.iteration = 1 #3
        self.dataset_number = 2 #3
        self.soil = [1, 5, 10]#, 10
        self.mu = 18.25
        self.sigma = 4.7
        self.max_tries = 1000
        self.names = {
            "active_learning_split": ["flower", "white_obj"],
            "only_flowers_split": ["flower"],
            "active_learning_bg": ["flower", "white_obj", "soil"],
            "only_flowers_bg": ["flower", "soil"]
        }
        self.id_backgroung = {
            "active_learning": 2,
            "only_flowers": 1
        }

        # nombre de classes associé
        self.nc = {k: len(v) for k, v in self.names.items()}
        self.imgsz = 128
        self.epochs = 10
        self.batch = 32
        self.patience = 25
