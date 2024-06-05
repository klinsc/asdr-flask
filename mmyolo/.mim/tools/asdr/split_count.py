# count classes of val.json and train.json in coco format dataset

import argparse
import json

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="COCO format dataset split counter")
    parser.add_argument(
        "--dir", type=str, default=".", help="path to the directory of the dataset"
    )
    parser.add_argument(
        "--train", type=str, default="train.json", help="train dataset in coco format"
    )
    parser.add_argument(
        "--val", type=str, default="val.json", help="val dataset in coco format"
    )
    return parser.parse_args()


def main(args):
    with open(args.dir + "/" + args.train, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        annotations = coco["annotations"]
        categories = coco["categories"]

        annotation_categories = list(map(lambda a: int(a["category_id"]), annotations))

        # remove classes that has only one sample, because it can't be split into the training and testing sets
        annotation_categories = list(
            filter(lambda i: annotation_categories.count(i) > 1, annotation_categories)
        )

        # count classes
        class_count = {}
        for i in annotation_categories:
            class_count[i] = class_count.get(i, 0) + 1

        print("train.json")
        print("class count:", len(class_count))
        print("class distribution:", class_count)
        print("total number of samples:", len(annotation_categories))
        print()

    with open(args.dir + "/" + args.val, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        annotations = coco["annotations"]
        categories = coco["categories"]

        annotation_categories = list(map(lambda a: int(a["category_id"]), annotations))

        # remove classes that has only one sample, because it can't be split into the training and testing sets
        annotation_categories = list(
            filter(lambda i: annotation_categories.count(i) > 1, annotation_categories)
        )

        # count classes
        class_count = {}
        for i in annotation_categories:
            class_count[i] = class_count.get(i, 0) + 1

        print("val.json")
        print("class count:", len(class_count))
        print("class distribution:", class_count)
        print("total number of samples:", len(annotation_categories))
        print()


# save classes distribution to a csv, seperate train and val, use class name instead of class id, order by number of samples ascending, and plot it, save it to pngs
def save_class_distribution(args):
    with open(args.dir + "/" + args.train, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        annotations = coco["annotations"]
        categories = coco["categories"]

        annotation_categories = list(map(lambda a: int(a["category_id"]), annotations))

        # remove classes that has only one sample, because it can't be split into the training and testing sets
        annotation_categories = list(
            filter(lambda i: annotation_categories.count(i) > 1, annotation_categories)
        )

        # count classes & % of total
        class_count = {}
        for i in annotation_categories:
            class_count[i] = class_count.get(i, 0) + 1

        total = len(annotation_categories)
        class_count_percent = {}
        for key in class_count.keys():
            class_count_percent[key] = class_count[key] / total

        # save to csv
        with open(args.dir + "/" + "train_class_distribution.csv", "w") as f:
            for key in class_count.keys():
                f.write(
                    "%s,%s,%s\n"
                    % (
                        categories[key]["name"],
                        class_count[key],
                        class_count_percent[key],
                    )
                )

    with open(args.dir + "/" + args.val, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        annotations = coco["annotations"]
        categories = coco["categories"]

        annotation_categories = list(map(lambda a: int(a["category_id"]), annotations))

        # remove classes that has only one sample, because it can't be split into the training and testing sets
        annotation_categories = list(
            filter(lambda i: annotation_categories.count(i) > 1, annotation_categories)
        )

        # count classes & % of total
        class_count = {}
        for i in annotation_categories:
            class_count[i] = class_count.get(i, 0) + 1

        total = len(annotation_categories)
        class_count_percent = {}
        for key in class_count.keys():
            class_count_percent[key] = class_count[key] / total

        # save to csv
        with open(args.dir + "/" + "val_class_distribution.csv", "w") as f:
            for key in class_count.keys():
                f.write(
                    "%s,%s,%s\n"
                    % (
                        categories[key]["name"],
                        class_count[key],
                        class_count_percent[key],
                    )
                )

    # plot
    with open(args.dir + "/" + "train_class_distribution.csv", "r") as f:
        lines = f.readlines()
        lines = list(map(lambda l: l.strip().split(","), lines))
        lines = sorted(lines, key=lambda l: int(l[1]))

        x = list(map(lambda l: l[0], lines))
        y = list(map(lambda l: int(l[1]), lines))

        plt.figure(figsize=(20, 10))
        plt.bar(x, y)
        plt.xticks(rotation=90)
        plt.xlabel("class name")
        plt.ylabel("number of samples")
        plt.title("train class distribution")
        plt.savefig(args.dir + "/" + "train_class_distribution.png")

    with open(args.dir + "/" + "val_class_distribution.csv", "r") as f:
        lines = f.readlines()
        lines = list(map(lambda l: l.strip().split(","), lines))
        lines = sorted(lines, key=lambda l: int(l[1]))

        x = list(map(lambda l: l[0], lines))
        y = list(map(lambda l: int(l[1]), lines))

        plt.figure(figsize=(20, 10))
        plt.bar(x, y)
        plt.xticks(rotation=90)
        plt.xlabel("class name")
        plt.ylabel("number of samples")
        plt.title("val class distribution")
        plt.savefig(args.dir + "/" + "val_class_distribution.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    save_class_distribution(args)
