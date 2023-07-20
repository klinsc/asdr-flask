from io import BytesIO
import os
from pdf2image import convert_from_bytes
from flask import Flask, request, make_response
from flask_cors import CORS
from yolov5 import YoloV5

# create a flask server to receive the pdf file and convert it to images and send it back to the client
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/upload', methods=['POST'])
def upload():
    try:
        print(request)
        # get the pdf file from the request
        file = request.files['files[]']
        # convert the pdf file to images
        images = convert_from_bytes(file.read(), dpi=300, fmt='jpeg')
        # create a buffer to store the images
        buffer = BytesIO()
        # save the images to the buffer
        images[0].save(buffer, format='JPEG')
        # get the buffer as a byte array
        byte_arr = buffer.getvalue()

        # return images to the client
        return make_response(byte_arr, 200, {'Content-Type': 'image/jpeg'})

    except Exception as e:
        print(e)

        return make_response('Internal Server Error', 500)


@app.route('/predict', methods=['POST'])
def predict():
    # choose the output format from the request query string
    output = request.args.get('type')

    try:
        # get the images from the request
        file = request.files['files[]']
        # convert the images to a byte array
        byte_arr = (file.read())

        # save image to disk
        with open('temp.jpg', 'wb') as f:
            f.write(byte_arr)

        # create a yolov5 object
        yolo = YoloV5()

        # predict the bounding boxes
        results = yolo.predict('temp.jpg')

        # remove the image from disk
        os.remove('temp.jpg')

        if output == 'json':
            # read the json file
            with open('results.json', 'r') as f:
                jsonFile = f.read()

                # return the json file to the client
                return make_response(jsonFile, 200, {'Content-Type': 'application/json'})

        # read the csv file
        with open('results.csv', 'r') as f:
            csvFile = f.read()

            # return the csv file to the client
            return make_response(csvFile, 200, {'Content-Type': 'text/csv'})

    except Exception as e:
        print(e)

        return make_response('Internal Server Error', 500)


if __name__ == '__main__':
    app.run(debug=True)
