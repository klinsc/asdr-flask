import torch


class YoloV5:
    def __init__(self):
        # Load the model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='assets/best.pt')  # default

    # Function to predict the bounding boxes
    def predict(self, image_name=None):
        # Inference
        results = self.model(f"{image_name}.jpg")

        # Results
        results.print()

        return results
