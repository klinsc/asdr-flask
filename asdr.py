import asyncio
import os
import time
from io import BytesIO

import pandas as pd
from flask import Flask, make_response, request
from flask_cors import CORS
from pdf2image.pdf2image import convert_from_bytes

from handle_component import HandleComponent
from handle_image import HandleImage
from prisma import Prisma
from yolov5 import YoloV5

os.environ["PRISMA_HOME_DIR "] = "/var/tmp"
os.environ["PRISMA_BINARY_CACHE_DIR"] = "/var/tmp"

# create a flask server to receive the pdf file and convert it to images and send it back to the client
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def index():
    return "asdr-flask-server"


@app.route("/health")
async def health():
    # check if the server is connected to the database
    try:
        prisma = Prisma()
        await prisma.connect()
        await prisma.disconnect()
    except Exception as e:
        print(e)
        return make_response(f"Internal Server Error: {e}", 500)

    return "OK"


@app.route("/upload", methods=["POST"])
def upload():
    try:
        # get the pdf file from the request
        file = request.files["image"]
        # convert the pdf file to images
        images = convert_from_bytes(file.read(), dpi=300, fmt="jpeg")
        # save the images to disk
        image = images[0]
        image.save(f"images/image.jpeg", "JPEG")

    except Exception as e:
        print(e)

        return make_response(f"Internal Server Error: {e}", 500)

    try:
        # read the image from disk
        image = None
        with open("images/image.jpeg", "rb") as f:
            image = f.read()

        if not image:
            return make_response("Internal Server Error: image not found", 500)

        # remove the image from disk
        os.remove("images/image.jpeg")

        # return images to the client
        return make_response(image, 200, {"Content-Type": "image/jpeg"})

    except Exception as e:
        print(e)

        return make_response(f"Internal Server Error: {e}", 500)


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        # get the drawing type id from the request
        drawing_type_id = request.args.get("drawingTypeId")
        if not drawing_type_id:
            return make_response("Bad Request: drawingTypeId is required", 400)

        # get the images from the request
        file = request.files["image"]

        # convert the images to a byte array
        byte_arr: bytearray = file.read()
        if not byte_arr:
            return make_response("Bad Request: file is required", 400)

        # generate a random name for the image
        image = HandleImage(byte_arr)
        image_path = image.get_image_path()

        # create a yolov5 object
        yolo = YoloV5()
        # predict the bounding boxes
        results = yolo.predict(image_path)
        # remove the image from disk
        image.remove()
        # create a df from the results
        df: pd.DataFrame = results.pandas().xyxy[0]

        # create a component handler
        component_handler = HandleComponent(df, drawing_type_id)
        # 1) add column id & color to drawing_components_df with value of id & color from database
        predicted_components_df = asyncio.run(component_handler.get_detail_components())
        if predicted_components_df is None:
            raise Exception("Error in get detail components")

        # 2) diagnose the components
        (
            found_components_df,
            remaining_components_df,
            missing_components_df,
        ) = asyncio.run(component_handler.diagnose_components())
        if (
            found_components_df is None
            or remaining_components_df is None
            or missing_components_df is None
        ):
            raise Exception("Error in diagnose components")

        # 3) sort the line type components
        sorted_line_type_components_df = component_handler.sort_line_type_components(
            found_components_df
        )
        if sorted_line_type_components_df is None:
            raise Exception("Error in sort line type components")

        # validate that found_components_df + remaining_components_df = predicted_components_df
        if len(predicted_components_df) != len(sorted_line_type_components_df) + len(
            remaining_components_df
        ):
            raise Exception("Error in sort: found + remaining != predicted")

        # 4) cluster the components
        clustered_found_components_df = component_handler.get_clustered_components(
            sorted_line_type_components_df
        )
        if clustered_found_components_df is None:
            raise Exception("Error in cluster components")

        # 5) get the hulls of the clustered components
        clustered_hulls = asyncio.run(
            component_handler.get_clustered_convexhull(clustered_found_components_df)
        )

        # 6) correct missing components
        corrected_missing_components_df = asyncio.run(
            component_handler.correct_missing_component(
                clustered_found_components_df, missing_components_df, clustered_hulls
            )
        )

        # return all dfs to the client in json format
        predicted_components_json = predicted_components_df.to_json(orient="records")
        found_components_json = found_components_df.to_json(orient="records")
        missing_components_json = corrected_missing_components_df.to_json(
            orient="records"
        )
        remaining_components_json = remaining_components_df.to_json(orient="records")
        clustered_found_components_json = clustered_found_components_df.to_json(
            orient="records"
        )
        clustered_hulls = clustered_hulls.to_json(orient="records")

        response = make_response(
            {
                "predicted_components": predicted_components_json,
                "found_components": found_components_json,
                "remaining_components": remaining_components_json,
                "missing_components": missing_components_json,
                "hulls": clustered_hulls,
                "clustered_found_components": clustered_found_components_json,
                "status": "success",
            },
            200,
            {"Content-Type": "application/json"},
        )

        return response
    except Exception as e:
        print(e)

        return make_response(f"Internal Server Error: {e}", 500)
    finally:
        print(f"predict() {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
