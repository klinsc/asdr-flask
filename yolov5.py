import torch


class YoloV5:
    def __init__(self):
        # Load the model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='assets/best.pt')  # default

    # Function to predict the bounding boxes
    def predict(self, imgs=None,output_type='csv'):
        # Inference
        results = self.model(imgs)

        # Results
        results.print()

        # Results as JSON
        if output_type == 'json':
            results.pandas().xyxy[0].to_json('results.json', orient='records')
            return 

        # Results as CSV
        results.pandas().xyxy[0].to_csv('results.csv', index=True)
        return 