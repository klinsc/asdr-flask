import asyncio
import json
import os
import time
import uuid
from io import BytesIO

import pandas as pd
from flask import Flask, after_this_request, make_response, request
from flask_cors import CORS
from pdf2image.pdf2image import convert_from_bytes

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


@app.route("/predict", methods=["POST"])
async def predict():
    @after_this_request
    def delete_results(response):
        try:
            os.remove(f"{image_name}.{out_type}")
        except Exception as ex:
            print(ex)
        return response

    # choose the output format from the request query string
    output = request.args.get("type")
    out_type = "csv" if output == "csv" else "json"

    try:
        # get the images from the request
        file = request.files["files[]"]
        # convert the images to a byte array
        byte_arr = file.read()

        # generate a random name for the image
        image_name = str(uuid.uuid4())

        # save image to disk
        with open(f"{image_name}.jpg", "wb") as f:
            f.write(byte_arr)

        # create a yolov5 object
        yolo = YoloV5()

        # predict the bounding boxes
        results = yolo.predict(f"{image_name}")

        # remove the image from disk
        os.remove(f"{image_name}.jpg")

        if output == "json":
            prisma = Prisma()
            await prisma.connect()

            newResults = results.pandas().xyxy[0].copy()

            # for each bounding box find its color in database
            for bounding_box in newResults.itertuples():
                # queries the component table on the database
                component = await prisma.component.find_first(
                    where={"name": bounding_box.name}
                )
                if component == None:
                    continue

                # add the color to the dataframe
                newResults.at[bounding_box.Index, "color"] = component.color

                # add the id to the dataframe
                newResults.at[bounding_box.Index, "id"] = component.id

            # close the database connection
            await prisma.disconnect()

            # # for each bounding box, generate a small uuid for each
            # newResults["id"] = newResults.apply(
            #     lambda row: str(uuid.uuid4())[:8], axis=1
            # )

            # Results as JSON
            newResults.to_json(f"{image_name}.json", orient="records")

            # read the json file
            with open(f"{image_name}.json", "r") as f:
                jsonFile = f.read()

                # create a response object
                response = make_response(
                    jsonFile, 200, {"Content-Type": "application/json"}
                )

                # remove the json file from disk
                delete_results(response)

                # return the json file to the client
                return response

        # Results as CSV
        results.pandas().xyxy[0].to_csv(f"{image_name}.csv", index=True)

        # read the csv file
        with open(f"{image_name}.csv", "r") as f:
            csvFile = f.read()

            # create a response object
            response = make_response(csvFile, 200, {"Content-Type": "text/csv"})

            # remove the csv file from disk
            delete_results(response)

            # return the csv file to the client
            return response

    except Exception as e:
        print(e)

        return make_response("Internal Server Error", 500)


async def handle_json_result(raw_json_result):
    time_start = time.time()
    try:
        # parse the json result
        parsed_json_result = json.loads(raw_json_result)

        # gives color for each bounding box
        prisma = Prisma()
        await prisma.connect()

        # use enumerate to speed up the loop
        for index, bounding_box in enumerate(parsed_json_result):
            # queries the component table on the database
            component = await prisma.component.find_first(
                where={"name": bounding_box["name"]}
            )
            if component == None:
                continue

            # add the color to the dataframe
            parsed_json_result[index]["color"] = component.color

            # add the id to the dataframe
            parsed_json_result[index]["id"] = component.id

        # close the database connection
        await prisma.disconnect()

        # convert the json result to string
        string_json_result = json.dumps(parsed_json_result)

        return string_json_result

    except Exception as e:
        print(e)
        return None

    finally:
        print(f"---handle_json_result() {time.time() - time_start} seconds ---")


async def validate_predicted_components(
    predicted_components_df,
):
    time_start = time.time()

    # database:
    prisma = Prisma()
    await prisma.connect()

    try:
        # get all components from the database, where the ComponentVersion selected===True
        componentversion = await prisma.componentversion.find_many(
            where={"selected": True}
        )
        if len(componentversion) == 0:
            raise Exception("Selected component version not found")
        components = await prisma.component.find_many(
            where={"componentVersionId": componentversion[0].id}
        )
        if len(components) == 0:
            raise Exception("Component not found")

        # for each component in the predicted components, check if all component names exist in components
        for row in predicted_components_df.iterrows():
            if row["name"] not in components:
                raise Exception("Some components not found")

        return None

    except Exception as e:
        print(e)
        return e

    finally:
        # close the database connection
        await prisma.disconnect()
        print(
            f"---validate_predicted_components() {time.time() - time_start} seconds ---"
        )


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
                            name = line_type_component.Component.name  # type: ignore

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
                                            "id": [
                                                line_type_component.Component.id  # type: ignore
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

        # queries the component table on the database
        components = await prisma.component.find_many()
        if len(components) == 0:
            raise Exception("Component not found")

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

        return drawing_components_df

    finally:
        print(f"---getIdComponents() {time.time() - time_start} seconds ---")


@app.route("/test-predict", methods=["POST"])
def test_predict():
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

        # validate the predicted components
        asyncio.run(validate_predicted_components(predicted_components_df))

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
