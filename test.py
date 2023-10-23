import unittest

import requests


class FlaskTestCase(unittest.TestCase):
    def test_predict_file(self):
        url = "http://localhost:5000/predict?drawingTypeId=clnu2kk3n001g34vklxtyrt4z"
        image_path = "./images/mrb-Manorom2-sm-bh.jpg"  # replace with your image path
        headers = {
            "enctype": "multipart/form-data",
            "ngrok-skip-browser-warning": "true",
        }
        with open(image_path, "rb") as image_file:
            # create a form data object
            form_data = {"files[]": image_file}

            # add the form data to the request
            response = requests.post(url, headers=headers, files=form_data)

        json_response = response.json()
        self.assertEqual(json_response["status"], "success")


if __name__ == "__main__":
    unittest.main()
