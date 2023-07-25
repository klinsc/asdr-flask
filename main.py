from io import BytesIO
import os
import uuid
from pdf2image import convert_from_bytes
from flask import Flask, after_this_request, request, make_response
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
    @after_this_request
    def delete_results(response):
        try:
            os.remove(f"{image_name}.{out_type}")
        except Exception as ex:
            print(ex)
        return response
    
    # choose the output format from the request query string
    output = request.args.get('type')
    out_type = 'csv' if output == 'csv' else 'json'

    try:
        # get the images from the request
        file = request.files['files[]']
        # convert the images to a byte array
        byte_arr = (file.read())

        # generate a random name for the image
        image_name = str(uuid.uuid4())

        # save image to disk
        with open(f'{image_name}.jpg', 'wb') as f:
            f.write(byte_arr)

        # create a yolov5 object
        yolo = YoloV5()

        # predict the bounding boxes
        results = yolo.predict(f'{image_name}')

        # remove the image from disk
        os.remove(f'{image_name}.jpg')

        if output == 'json':
            # Results as JSON
            results.pandas().xyxy[0].to_json(
                f'{image_name}.json', orient='records')

            # read the json file
            with open(f'{image_name}.json', 'r') as f:
                jsonFile = f.read()

                # create a response object
                response =  make_response(jsonFile, 200, {'Content-Type': 'application/json'})

                # remove the json file from disk
                delete_results(response)

                # return the json file to the client
                return response

        # Results as CSV
        results.pandas().xyxy[0].to_csv(f'{image_name}.csv', index=True)

        # read the csv file
        with open(f'{image_name}.csv', 'r') as f:
            csvFile = f.read()

            # create a response object
            response =  make_response(csvFile, 200, {'Content-Type': 'text/csv'})

            # remove the csv file from disk
            delete_results(response)

            # return the csv file to the client
            return response
    
    except Exception as e:
        print(e)

        return make_response('Internal Server Error', 500)


if __name__ == '__main__':
    app.run(debug=True)
