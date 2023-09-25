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
            )

            # self.error = False
            # self.error_message = None
            self.error = {"error": False, "error_message": None}
        # catch exception if model not found
        except Exception as e:
            print(e)
            self.error = {"error": True, "error_message": "Error loading model"}

    # Function to predict the bounding boxes
    def predict(self, image_name=None):
        # Inference
        results = self.model(f"{image_name}.jpg", size=1280)

        # Results
        results.print()

        return results
