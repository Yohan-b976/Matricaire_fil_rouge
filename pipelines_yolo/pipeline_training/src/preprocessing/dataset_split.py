import random
import shutil
from pathlib import Path


def split_dataset(src, out,
                  train=0.8, val=0.2, test=0,
                  ext=".png", seed=42):

    assert abs(train + val + test - 1.0) < 1e-6
    assert Path(src).exists()

    src = Path(src)
    out = Path(out)

    random.seed(seed)

    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = sorted(src.glob(f"*{ext}"))

    valid_images = []
    ignored = 0

    for img in images:
        if img.with_suffix(".txt").exists():
            valid_images.append(img)
        else:
            ignored += 1

    images = valid_images
    print(src)

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train)
    n_val = int(n_total * val)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    def copy_pairs(img_list, split):
        for img in img_list:
            shutil.copy(img, out / "images" / split / img.name)
            shutil.copy(img.with_suffix(".txt"),
                        out / "labels" / split / img.with_suffix(".txt").name)

    copy_pairs(train_imgs, "train")
    copy_pairs(val_imgs, "val")
    copy_pairs(test_imgs, "test")

    print(f"Split terminé → {out}")
    print(f"Train : {len(train_imgs)} images")
    print(f"Val   : {len(val_imgs)} images")
    print(f"Test  : {len(test_imgs)} images")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--ext", default=".png")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    split_dataset(**vars(args))
