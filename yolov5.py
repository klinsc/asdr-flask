import torch


class YoloV5:
    def __init__(self):
        # Load the model
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path="./models/best_dc4c535796dc43b88f23fcc811b17a4c.pt",
        )  # default

    # Function to predict the bounding boxes
    def predict(self, image_name=None):
        # Inference
        results = self.model(f"{image_name}.jpg", size=1280)

        # Results
        results.print()

        return results
