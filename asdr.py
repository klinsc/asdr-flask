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
from inference import InferenceMMDet
from prisma import Prisma

# *deprecated
# from yolov5 import YoloV5

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

        # get the name of an image
        image_name = file.filename.split(".")[0]  # type: ignore

        # save the images to disk
        image = images[0]
        image.save(f"images/{image_name}.jpeg", "JPEG")

        with open(f"images/{image_name}.jpeg", "rb") as f:
            image = f.read()

        if not image:
            return make_response("Internal Server Error: image not found", 500)

        # remove the image from disk
        os.remove(f"images/{image_name}.jpeg")

        # return images to the client
        return make_response(image, 200, {"Content-Type": "image/jpeg"})

    except Exception as e:
        print(e)

        return make_response(f"Internal Server Error: {e}", 500)


@app.route("/predict", methods=["POST"])
def predict(
    debug=False,
    image_path: str | None = None,
    drawing_type_id: str | None = None,
    file_name: str | None = None,
):
    start_time = time.time()
    try:
        if not debug:
            # get the drawing type id from the request
            drawing_type_id = request.args.get("drawingTypeId")
            if not drawing_type_id:
                return make_response("Bad Request: drawingTypeId is required", 400)

            file_name = request.args.get("fileName")
            if not file_name:
                return make_response("Bad Request: fileName is required", 400)

            # get the images from the request
            file = request.files["image"]

            # convert the images to a byte array
            byte_arr: bytearray = file.read()
            if not byte_arr:
                return make_response("Bad Request: file is required", 400)

            # generate a random name for the image
            file_name = file_name.split(".")[0]  # type: ignore
            image = HandleImage(byte_arr, file_name)
            image_path = image.get_image_path()

        if not image_path:
            return make_response("Internal Server Error: image path not found", 500)

        if not drawing_type_id:
            return make_response(
                "Internal Server Error: drawing type id not found", 500
            )

        # *deprecated
        # create a yolov5 object
        # yolo = YoloV5()
        # results = yolo.predict(image_path)
        # create a df from the results
        # df: pd.DataFrame = results.pandas().xyxy[0]

        # create a mmdet object
        config_path = "configs/yolov5_s-p6-v62_syncbn_fast_4xb8-300e_asdr6-3-1000oa-split8020_1280_anchOptm_oPipe.py"
        checkpoint_path = "checkpoints/yolov5_s-p6-v62_syncbn_fast_4xb8-300e_asdr6-3-1000oa-split8020_1280_anchOptm_oPipe.pth"
        model = InferenceMMDet(config_path, checkpoint_path)

        # predict the bounding boxes
        df = pd.DataFrame()
        if not debug:
            df = model.inference(image_path)
        else:
            debug_name = image_path.split("/")[-1].split(".")[0]
            if not os.path.exists(f"temp/{debug_name}.csv"):
                df = model.inference(image_path)
                df.to_csv(f"temp/{debug_name}.csv", index=False)
            else:
                df = pd.read_csv(f"temp/{debug_name}.csv")

        # create a component handler
        # debug = True
        component_handler = HandleComponent(True, df, drawing_type_id, image_path)

        # 1) add column id & color to drawing_components_df with value of id & color from database
        predicted_components_df = asyncio.run(component_handler.get_detail_components())
        if predicted_components_df is None:
            raise Exception("Error in get detail components")

        # 1.1) fill black color on the image
        asyncio.run(component_handler.fill_black_color(predicted_components_df))

        # 2) diagnose the components
        (
            found_components_df,
            remaining_components_df,
            missing_components_df,
        ) = asyncio.run(component_handler.diagnose_components(image_path))
        # ) = asyncio.run(component_handler.diagnose_components_v2(image_path, file_name))
        if (
            found_components_df is None
            or remaining_components_df is None
            or missing_components_df is None
        ):
            raise Exception("Error in diagnose components")

        # component_handler.display(
        #     found_components_df,
        #     remaining_components_df,
        #     missing_components_df,
        #     image_path,
        # )

        found_components_df = asyncio.run(
            component_handler.assign_cluster_number(
                found_components_df,
            )
        )
        if found_components_df is None:
            raise Exception("Error in assign cluster number")

        # component_handler.display_cluster(
        #     found_components_df,
        #     remaining_components_df,
        #     missing_components_df,
        #     image_path,
        # )

        # 5.1) get the hulls of the found components
        found_component_hulls = asyncio.run(
            component_handler.get_found_convexhull(found_components_df)
        )

        # 6) correct missing components (The modification of correcting is directly affect to missing_components_df)
        asyncio.run(
            component_handler.correct_missing_component(
                clustered_found_components_df, missing_components_df, clustered_hulls
            )
        )

        # return all dfs to the client in json format
        predicted_components_json = predicted_components_df.to_json(orient="records")
        found_components_json = found_components_df.to_json(orient="records")
        missing_components_json = missing_components_df.to_json(orient="records")
        remaining_components_json = remaining_components_df.to_json(orient="records")
        clustered_found_components_json = clustered_found_components_df.to_json(
            orient="records"
        )
        clustered_hulls = clustered_hulls.to_json(orient="records")
        found_component_hulls = found_component_hulls.to_json(orient="records")

        response = make_response(
            {
                "predicted_components": predicted_components_json,
                "found_components": found_components_json,
                "remaining_components": remaining_components_json,
                "missing_components": missing_components_json,
                "hulls": clustered_hulls,
                "clustered_found_components": clustered_found_components_json,
                "found_component_hulls": found_component_hulls,
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
        if not debug and image_path:
            # remove the image from disk
            os.remove(image_path)

        print(f"predict() {time.time() - start_time} seconds ---")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="ASDR Flask Server")
    parser.add_argument(
        "--debug",
        type=str,
        default="false",
        help="Run the server in debug mode",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="images/image.jpeg",
        help="The path to the image",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug == "true":
        predict(
            debug=True,
            image_path=args.image_path,
            drawing_type_id="clwx2950z0000348civ5vsm84",
        )

    else:
        app.run(debug=False, host="0.0.0.0")


if __name__ == "__main__":
    main()
