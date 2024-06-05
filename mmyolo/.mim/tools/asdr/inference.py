# Description: inference for asdr, take config, checkpoint, image as input, return result

import argparse
import os

import cv2
import mmcv
from matplotlib import pyplot as plt
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


def args_parser():
    parser = argparse.ArgumentParser(description="inference for asdr")
    parser.add_argument("--config", type=str, help="config file path")
    parser.add_argument("--checkpoint", type=str, help="checkpoint file path")
    parser.add_argument("--image", type=str, help="image file path")
    parser.add_argument(
        "--threshold", type=float, help="threshold for nms", default=0.3
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device used for inference",
        required=False,
    )
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    image = mmcv.imread(args.image)
    result = inference_detector(
        model,
        image,
    )

    # init the visualizer(execute this block only once)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # get config name
    config_name = args.config.split("/")[-1]
    # without .py
    config_name = config_name.split(".")[0]

    # get filename
    filename = args.image.split("/")[-1]
    # without .(jpg|png|jpeg)
    filename = filename.split(".")[0]
    # add threshold
    filename = f"{filename}_{args.threshold}"

    # save path
    save_path = f"outputs/{os.getenv('MMYOLO_OUTPUT')}"

    # check if save path exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # check if the same filename exists in the save path
    if os.path.exists(f"{save_path}/{filename}.jpg"):
        # increment the filename
        i = 1
        while os.path.exists(f"{save_path}/{filename}({i}).jpg"):
            i += 1
        filename = f"{filename}({i})"

    # show the results & save the results
    visualizer.add_datasample(
        "result",
        image,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
        out_file=f"{save_path}/{filename}.jpg",
        pred_score_thr=args.threshold,
    )
    visualizer.show()


if __name__ == "__main__":
    main()
