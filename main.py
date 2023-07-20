from io import BytesIO
from pdf2image import convert_from_bytes
from flask import Flask, request, make_response
from flask_cors import CORS


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


if __name__ == '__main__':
    app.run(debug=True)
