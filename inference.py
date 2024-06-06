# Description: inference for asdr, take config, checkpoint, image as input, return result

import os
import uuid

import mmcv
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
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
                    "x_center": (bbox_result[:, 0] + bbox_result[:, 2]) / 2,
                    "y_center": (bbox_result[:, 1] + bbox_result[:, 3]) / 2,
                    # use "confidence" instead of "score" to implement with asdr-flask
                    "confidence": score_result,
                    "class": label_result,
                    "predicted_id": [
                        str(uuid.uuid4())[:8] for _ in range(len(bbox_result))
                    ],
                }
            )
            class_names = self.model.dataset_meta["classes"]  # type: ignore
            dataframe_result["name"] = dataframe_result["class"].apply(
                lambda x: class_names[int(x)]  # type: ignore
            )  # type: ignore

            # filter with score > 0.5
            dataframe_result = dataframe_result[dataframe_result["confidence"] > 0.5]

            # if components are reside in the same area, overlap percentage > 0.5
            # with the same name, then they are the same component
            # remove the duplicate components, keep the first one that has the highest confidence
            remaining_components_df = dataframe_result.sort_values(
                by=["xmin", "ymin", "xmax", "ymax", "name"],
                ascending=[True, True, False, False, False],
            )
            # find overlapping components
            overlapping_components = []
            for i in range(len(remaining_components_df)):
                for j in range(i + 1, len(remaining_components_df)):
                    # if the two components are the same
                    if (
                        remaining_components_df.iloc[i]["name"]
                        == remaining_components_df.iloc[j]["name"]
                    ):
                        component_i = remaining_components_df.iloc[i]
                        component_j = remaining_components_df.iloc[j]

                        # if i center is in j
                        if (
                            remaining_components_df.iloc[i]["x_center"]
                            >= remaining_components_df.iloc[j]["xmin"]
                            and remaining_components_df.iloc[i]["x_center"]
                            <= remaining_components_df.iloc[j]["xmax"]
                            and remaining_components_df.iloc[i]["y_center"]
                            >= remaining_components_df.iloc[j]["ymin"]
                            and remaining_components_df.iloc[i]["y_center"]
                            <= remaining_components_df.iloc[j]["ymax"]
                        ):
                            overlapping_components.append(component_i.predicted_id)
                        # if j center is in i
                        elif (
                            remaining_components_df.iloc[j]["x_center"]
                            >= remaining_components_df.iloc[i]["xmin"]
                            and remaining_components_df.iloc[j]["x_center"]
                            <= remaining_components_df.iloc[i]["xmax"]
                            and remaining_components_df.iloc[j]["y_center"]
                            >= remaining_components_df.iloc[i]["ymin"]
                            and remaining_components_df.iloc[j]["y_center"]
                            <= remaining_components_df.iloc[i]["ymax"]
                        ):
                            overlapping_components.append(component_j.predicted_id)

            # remove duplicate of overlapping_components
            overlapping_components = list(set(overlapping_components))

            # # plot the overlapping components
            # fig, ax = plt.subplots()
            # ax.imshow(image)
            # for id in overlapping_components:
            #     component = remaining_components_df[
            #         remaining_components_df["predicted_id"] == id
            #     ].iloc[0]
            #     rect = Rectangle(
            #         (component["xmin"], component["ymin"]),
            #         component["xmax"] - component["xmin"],
            #         component["ymax"] - component["ymin"],
            #         linewidth=1,
            #         edgecolor="r",
            #         facecolor="none",
            #     )
            #     ax.add_patch(rect)
            #     ax.text(
            #         component["xmin"],
            #         component["ymin"],
            #         f"{component['name']} {component['confidence']:.2f}",
            #         fontsize=8,
            #         color="r",
            #     )
            # plt.show()

            # create a new dataframe filter out the overlapping_components
            for pred_id in overlapping_components:
                # remaining_components_df = remaining_components_df[
                #     remaining_components_df["predicted_id"] != pred_id
                # ]
                remaining_components_df = remaining_components_df.drop(
                    remaining_components_df[
                        remaining_components_df["predicted_id"] == pred_id
                    ].index
                )

            # sort by x_center, y_center left to right, then top to bottom
            remaining_components_df = remaining_components_df.sort_values(
                by=["x_center", "y_center"],
                ascending=[True, True],
            )

            # reset the index
            remaining_components_df.reset_index(drop=True, inplace=True)

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

            return remaining_components_df

        except Exception as e:
            print(e)
            raise Exception(e)
