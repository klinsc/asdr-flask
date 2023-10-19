import os
import time

import torch


class YoloV5:
    def __init__(self):
        try:
            self.file_name = "best_dc4c535796dc43b88f23fcc811b17a4c.pt"
            assert self.file_name in os.listdir("./models"), "Model not found"

            # Load the model
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=f"./models/{self.file_name}",
                trust_repo=True,
            )

            # # Config IoU threshold
            # self.model.conf = 0.4
            # self.model.iou = 0.5

        # catch exception if model not found
        except Exception as e:
            print(e)
            raise Exception(e)

    # Function to predict the bounding boxes
    def predict(self, image_path=None, size=1280):
        start_time = time.time()
        try:
            assert image_path is not None, "An image not provided"

            # Inference
            results = self.model(image_path, size=size)
            return results

        # catch exception if image not found
        except Exception as e:
            print(e)
            raise Exception(e)

        finally:
            print(f"---predict() {time.time() - start_time} seconds ---")