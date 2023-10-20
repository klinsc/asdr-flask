import asyncio
import json
import os
import time
import uuid
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, after_this_request, make_response, request
from flask_cors import CORS
from pdf2image.pdf2image import convert_from_bytes
from PIL import Image
from scipy.spatial import ConvexHull, Voronoi, convex_hull_plot_2d, voronoi_plot_2d

from drawing_tree import drawing_tree
from prisma import Prisma  # type: ignore
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
        file = request.files["files[]"]
        # convert the pdf file to images
        images = convert_from_bytes(file.read(), dpi=300, fmt="jpeg")
        # create a buffer to store the images
        buffer = BytesIO()
        # save the images to the buffer
        images[0].save(buffer, format="JPEG")
        # get the buffer as a byte array
        byte_arr = buffer.getvalue()

        # return images to the client
        return make_response(byte_arr, 200, {"Content-Type": "image/jpeg"})

    except Exception as e:
        print(e)

        return make_response("Internal Server Error", 500)


def getLineTypeConvexHull(
    line_type_component: pd.DataFrame,
) -> tuple[list[dict[str, float]]]:
    """
    Returns the convex hull of the line type
    """
    time_start = time.time()
    try:
        # get center point of each line type component
        line_type_component["center_x"] = (
            line_type_component["xmin"] + line_type_component["xmax"]
        ) / 2
        line_type_component["center_y"] = (
            line_type_component["ymin"] + line_type_component["ymax"]
        ) / 2

        # get the convex hull of the line type
        hull = ConvexHull(line_type_component[["center_x", "center_y"]].values)

        # return the convex hull as json
        return (
            [
                {
                    "x": float(line_type_component["center_x"].values[i]),
                    "y": float(line_type_component["center_y"].values[i]),
                }
                for i in hull.vertices
            ],
        )

    except Exception as e:
        print(e)
        raise e

    finally:
        print(f"---getIdComponents() {time.time() - time_start} seconds ---")


def getFoundComponentsConvexHull(
    found_components_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns the convex hull of the found components
    """
    time_start = time.time()
    try:
        # group by line type id & atCount and generate new uuid for each group
        groups = found_components_df.groupby(["lineTypeId", "atCount"]).groups
        groups = pd.DataFrame(
            {
                "lineTypeId": [groupId[0] for groupId in groups],
                "atCount": [groupId[1] for groupId in groups],
                "key": [str(uuid.uuid4())[:8] for i in range(len(groups))],
            }
        )

        # create a new dataframe to store the hull of each line type id
        hulls = pd.DataFrame(columns=["lineTypeId", "hull", "key"])

        for index, row in groups.iterrows():
            # get the line type components of the line type id
            line_type_components = found_components_df[
                (found_components_df["lineTypeId"] == row["lineTypeId"])
                & (found_components_df["atCount"] == row["atCount"])
            ]

            # skip for line type components length < 3
            if len(line_type_components) < 3:
                continue

            # get the convex hull of the line type components
            hull = getLineTypeConvexHull(line_type_components)

            # add the hull to the hulls dataframe
            hulls = pd.concat(
                [
                    hulls,
                    pd.DataFrame(
                        {
                            "lineTypeId": [row["lineTypeId"]],
                            "hull": [hull],
                            "key": [row["key"]],
                        }
                    ),
                ],
                ignore_index=True,
            )

        return hulls

    except Exception as e:
        print(e)
        raise e

    finally:
        print(f"---getIdComponents() {time.time() - time_start} seconds ---")


async def diagnose_components(
    predicted_components_df: pd.DataFrame, drawing_type_id: str
):
    time_start = time.time()
    try:
        # database:
        prisma = Prisma()
        await prisma.connect()

        # query the database on drawing type table
        drawingType = await prisma.drawingtype.find_first(where={"id": drawing_type_id})
        if drawingType == None:
            raise Exception("Drawing type not found")

        # queries all lines in the database on line type table
        line_types = await prisma.linetype.find_many(
            where={"drawingTypeId": drawingType.id}, order={"index": "asc"}
        )
        if len(line_types) == 0:
            raise Exception("Line not found")

        # get all line types with their components
        line_types = await prisma.linetype.find_many(
            where={"drawingTypeId": drawingType.id},
            order={"index": "asc"},
            include={"LineTypeComponent": {"include": {"Component": True}}},
        )
        if len(line_types) == 0:
            raise Exception("Line type not found")

        # define: remaining components
        remaining_components_df = predicted_components_df.copy()

        # define: missing components
        missing_components_df = predicted_components_df.copy().drop(
            predicted_components_df.index
        )

        # define: found components
        found_components_df = predicted_components_df.copy().drop(
            predicted_components_df.index
        )

        # loop through line_types to diagnose the LineTypeComponent
        for line_type in line_types:
            for i in range(line_type.count):
                # loop through the LineTypeComponents of the line type
                for line_type_component in line_type.LineTypeComponent:  # type: ignore
                    for i in range(line_type_component.count):
                        if (
                            line_type_component.Component.name  # type: ignore
                            in remaining_components_df["name"].values
                        ):
                            # get the first index of the component in the remaining components
                            index = remaining_components_df[
                                remaining_components_df["name"]
                                == line_type_component.Component.name  # type: ignore
                            ].index[0]

                            # add the component to the found components
                            found_components_df = pd.concat(
                                [
                                    found_components_df,
                                    remaining_components_df.loc[[index]],
                                ],
                                ignore_index=True,
                            )

                            # also add the lineTypeId to the recently found component
                            last_index = len(found_components_df) - 1
                            found_components_df.at[
                                last_index, "lineTypeId"
                            ] = line_type_component.lineTypeId

                            # also add count number to the recently found component
                            found_components_df.at[last_index, "atCount"] = i + 1

                            # remove the component from the remaining components
                            remaining_components_df.drop(index, inplace=True)

                        # add the component to the missing components
                        else:
                            missing_components_df = pd.concat(
                                [
                                    missing_components_df,
                                    pd.DataFrame(
                                        {
                                            "name": [
                                                line_type_component.Component.name  # type: ignore
                                            ],
                                            "color": [
                                                line_type_component.Component.color  # type: ignore
                                            ],
                                            "key": [
                                                str(uuid.uuid4())[
                                                    :8
                                                ]  # generate a small uuid
                                            ],
                                            "lineTypeId": [
                                                line_type_component.lineTypeId
                                            ],
                                        }
                                    ),
                                ],
                                ignore_index=True,
                            )

        # close the database connection
        await prisma.disconnect()

        return found_components_df, remaining_components_df, missing_components_df

    except Exception as e:
        print(e)
        return None, None, None

    finally:
        print(f"---diagnose_components() {time.time() - time_start} seconds ---")


async def getDetailComponents(drawing_components_df: pd.DataFrame):
    """
    Adds the componentId, color and key of the components to the dataframe
    """
    time_start = time.time()
    try:
        # database:
        prisma = Prisma()
        await prisma.connect()

        # get all components from the database, where the ComponentVersion selected===True
        componentversion = await prisma.componentversion.find_many(
            where={"selected": True}
        )
        if len(componentversion) == 0:
            raise Exception("Selected component version not found")

        # get all components from the database, where the ComponentVersion selected===True
        components = await prisma.component.find_many(
            where={"componentVersionId": componentversion[0].id}
        )
        if len(components) == 0:
            raise Exception("Component not found")

        # # for each component in the drawing_components_df, check if all component names exist in components
        if not all(
            drawing_components_df["name"].isin(
                [component.name for component in components]
            )
        ):
            raise Exception("Some components not found")

        for index, row in drawing_components_df.iterrows():
            for component in components:
                if row["name"] == component.name:
                    drawing_components_df.at[index, "componentId"] = component.id
                    drawing_components_df.at[index, "color"] = component.color
                    drawing_components_df.at[index, "key"] = str(uuid.uuid4())[:8]
                    break

        # close the database connection
        await prisma.disconnect()
        return drawing_components_df

    except Exception as e:
        print(e)
        raise e

    finally:
        print(f"---getIdComponents() {time.time() - time_start} seconds ---")


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        # get the drawing type id from the request
        drawing_type_id = request.args.get("drawingTypeId")
        if not drawing_type_id:
            return make_response("Bad Request: drawingTypeId is required", 400)

        # get the images from the request
        file = request.files["files[]"]
        # convert the images to a byte array
        byte_arr = file.read()

        # generate a random name for the image
        image_name = f"{str(uuid.uuid4())}.jpg"

        # check if the images folder exists
        if not os.path.exists("./images"):
            os.makedirs("./images")

        # create the image path
        image_path = f"./images/{image_name}"

        # save image to disk
        with open(image_path, "wb") as f:
            f.write(byte_arr)

        # create a yolov5 object
        yolo = YoloV5()

        # predict the bounding boxes
        results = yolo.predict(image_path)

        # remove the image from disk
        os.remove(image_path)

        # create a df from the results
        df: pd.DataFrame = results.pandas().xyxy[0]

        # copy from df to predicted_components_df with index
        predicted_components_df = df.copy(deep=True).reset_index()

        # add column id & color to drawing_components_df with value of id & color from database
        predicted_components_df = asyncio.run(
            getDetailComponents(predicted_components_df)
        )

        # diagnose the components
        (
            found_components_df,
            remaining_components_df,
            missing_components_df,
        ) = asyncio.run(diagnose_components(predicted_components_df, drawing_type_id))
        if (
            found_components_df is None
            or remaining_components_df is None
            or missing_components_df is None
        ):
            raise Exception("Error in diagnose components")

        # validate that found_components_df + remaining_components_df = predicted_components_df
        if len(predicted_components_df) != len(found_components_df) + len(
            remaining_components_df
        ):
            raise Exception("Error in diagnose: found + remaining != predicted")

        # get hulls
        hulls = getFoundComponentsConvexHull(found_components_df)
        hulls = hulls.to_json(orient="records")

        # return all dfs to the client in json format
        predicted_components_json = predicted_components_df.to_json(orient="records")
        found_components_json = found_components_df.to_json(orient="records")
        missing_components_json = missing_components_df.to_json(orient="records")
        remaining_components_json = remaining_components_df.to_json(orient="records")

        response = make_response(
            {
                "predicted_components": predicted_components_json,
                "found_components": found_components_json,
                "remaining_components": remaining_components_json,
                "missing_components": missing_components_json,
                "hulls": hulls,
            },
            200,
            {"Content-Type": "application/json"},
        )

        # # temp
        # response = make_response(string_json_result, 200, {"Content-Type": "json"})

        return response
    except Exception as e:
        print(e)

        return make_response("Internal Server Error", 500)
    finally:
        print(f"---test_predict() {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
