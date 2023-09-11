import torch


class YoloV5:
    def __init__(self):
        # Load the model
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path="./models/72bfd85d53b140938c8058c8fbaa362c.pt",
        )  # default

    # Function to predict the bounding boxes
    def predict(self, image_name=None):
        # Inference
        results = self.model(f"{image_name}.jpg")

        # Results
        results.print()

        return results
