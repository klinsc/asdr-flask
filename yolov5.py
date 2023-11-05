import os
import time

import torch


class YoloV5:
    def __init__(self):
        try:
            self.file_name = (
                "best_423fee3d1ee649bfa9e353184b4b3ddb.pt"  # 150e minClass100
            )
            # self.file_name = (
            #     "best_76d22b15ff974b95b731903c56d73c64.pt"  # 60e minClass500
            # )

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

    def is_locked(self, filepath):
        locked = None
        file_object = None
        if os.path.exists(filepath):
            try:
                buffer_size = 8
                # Opening file in append mode and read the first 8 characters.
                file_object = open(filepath, "a", buffer_size)
                if file_object:
                    locked = False
            except IOError as message:
                locked = True
            finally:
                if file_object:
                    file_object.close()
        return locked

    def wait_for_file(self, filepath):
        wait_time = 1
        while self.is_locked(filepath):
            time.sleep(wait_time)

    # Function to predict the bounding boxes
    def predict(self, image_path=None, size=1280):
        start_time = time.time()
        try:
            assert image_path is not None, "An image not provided"

            # Wait for file to be unlocked
            self.wait_for_file(image_path)

            # Inference
            results = self.model(image_path, size=size)
            return results

        # catch exception if image not found
        except Exception as e:
            print(e)
            raise Exception(e)

        finally:
            print(f"---predict() {time.time() - start_time} seconds ---")
