# display class distribution by area size from a dataset


import argparse

import cv2
import numpy as np
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(
        description="display class area size from a dataset in csv format, order ascendingly, and show its image_id too"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        required=True,
        help="path to the directory of the dataset",
    )
    parser.add_argument(
        "--ann",
        type=str,
        default=None,
        required=True,
        help="train/val annotation in coco format",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        required=True,
        help="class name to display",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./size.csv",
        help="output csv file path",
    )
    return parser.parse_args()


def main(args):
    coco = COCO(args.dir + "/" + args.ann)
    annotations = coco.loadAnns(coco.getAnnIds())
    categories = coco.loadCats(coco.getCatIds())
    category_names = list(map(lambda c: c["name"], categories))

    # get class id
    class_id = category_names.index(args.class_name)

    # get area for each annotation
    areas = list(map(lambda ann: ann["area"], annotations))

    # get image id for each annotation
    image_ids = list(map(lambda ann: ann["image_id"], annotations))

    # get class id for each annotation
    class_ids = list(map(lambda ann: ann["category_id"], annotations))

    # get image name for each image id
    image_names = list(map(lambda i: coco.loadImgs(i)[0]["file_name"], image_ids))

    # save to csv
    with open(args.out, "w") as f:
        f.write("image_id,area,class_id\n")
        for i in range(len(annotations)):
            if class_ids[i] == class_id:
                f.write(f"{image_ids[i]},{areas[i]},{class_ids[i]},{image_names[i]}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
