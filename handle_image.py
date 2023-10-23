import os
import uuid


class HandleImage:
    def __init__(self, image_byte_arr: bytearray):
        if image_byte_arr == None:
            raise Exception("Image byte array is required")

        if not os.path.exists("./images"):
            os.makedirs("./images")

        self.image_byte_arr = image_byte_arr
        self.image_name = f"{str(uuid.uuid4())}.jpg"
        self.image_path = f"./images/{self.image_name}"
        with open(self.image_path, "wb") as f:
            f.write(self.image_byte_arr)

    def remove(self):
        os.remove(self.image_path)

    def get_image_path(self):
        return self.image_path

    def get_image_name(self):
        return self.image_name
