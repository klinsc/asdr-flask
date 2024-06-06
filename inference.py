# Description: inference for asdr, take config, checkpoint, image as input, return result

import os

import mmcv
import pandas as pd
import torch
from matplotlib import pyplot as plt
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


class InferenceMMDet:
    def __init__(self, config_path, checkpoint_path):
        try:
            assert os.path.exists(config_path), "Config file not found"
            assert os.path.exists(checkpoint_path), "Checkpoint file not found"

            self.config_path = config_path
            self.checkpoint_path = checkpoint_path
            self.model = init_detector(config_path, checkpoint_path, device="cpu")

        except Exception as e:
            print(e)
            raise Exception(e)

    def inference(self, image_path):
        try:
            # Check if the image exists
            assert os.path.exists(image_path), "Image not found"

            # Load the image
            image = mmcv.imread(image_path)

            # Perform inference
            result = inference_detector(
                self.model,
                image,
            )

            # Extracting bounding boxes
            # The result is a list of arrays, one for each class.
            # Each array is of shape (n, 5) where n is the number of detected bounding boxes,
            # and the 5 columns are [x1, y1, x2, y2, score].
            # bbox_result = copy.deepcopy(result.pred_instances.bboxes).toTensor()
            bbox_result = torch.Tensor.cpu(result.pred_instances.bboxes)  # type: ignore
            score_result = torch.Tensor.cpu(result.pred_instances.scores)  # type: ignore
            label_result = torch.Tensor.cpu(result.pred_instances.labels)  # type: ignore
            dataframe_result = pd.DataFrame(
                {
                    "xmin": bbox_result[:, 0],
                    "ymin": bbox_result[:, 1],
                    "xmax": bbox_result[:, 2],
                    "ymax": bbox_result[:, 3],
                    # use "confidence" instead of "score" to implement with asdr-flask
                    "confidence": score_result,
                    "class": label_result,
                }
            )
            class_names = self.model.dataset_meta["classes"]  # type: ignore
            dataframe_result["name"] = dataframe_result["class"].apply(
                lambda x: class_names[int(x)]  # type: ignore
            )  # type: ignore

            # filter with score > 0.5
            dataframe_result = dataframe_result[dataframe_result["confidence"] > 0.5]

            # # print the result
            # print(dataframe_result)

            # # init the visualizer(execute this block only once)
            # visualizer = VISUALIZERS.build(self.model.cfg.visualizer)  # type: ignore
            # # the dataset_meta is loaded from the checkpoint and
            # # then pass to the model in init_detector
            # visualizer.dataset_meta = self.model.dataset_meta

            # # show the results & save the results
            # visualizer.add_datasample(
            #     "result",
            #     image,
            #     data_sample=result,
            #     draw_gt=False,
            #     wait_time=0,
            #     out_file=f"results.jpg",
            #     pred_score_thr=0.5,
            # )
            # visualizer.show()

            return dataframe_result

        except Exception as e:
            print(e)
            raise Exception(e)
