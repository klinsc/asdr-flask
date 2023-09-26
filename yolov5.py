import os

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

        # catch exception if model not found
        except Exception as e:
            print(e)
            raise Exception(e)

    # Function to predict the bounding boxes
    def predict(self, image_path=None, size=1280):
        try:
            assert image_path is not None, "An image not provided"

            # Inference
            results = self.model(image_path, size=size)
            return results

        # catch exception if image not found
        except Exception as e:
            print(e)
            raise Exception(e)
