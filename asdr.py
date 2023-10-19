import asyncio
import json
import os
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


async def validate_predicted_components(
    predicted_components_df,
):
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


async def diagnose_components(predicted_components_df, drawing_type_id):
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

        # # queries all lineTypeComnents in the database on line type component table
        # line_types_with_components = []
        # for line_type in line_types:
        #     components = await prisma.linetypecomponent.find_many(
        #         where={"lineTypeId": line_type.id},
        #         order={"index": "asc"},
        #         include={"Component": True},
        #     )

        #     # define new type for components
        #     new_components = []
        #     for component in components:
        #         new_component = {
        #             ...component.Component.as_dict(),
        #         }
        #         new_components.append(new_component)

        #     line_types_with_component = {
        #         "line_type": line_type,
        #         "components": components,
        #     }

        #     line_types_with_components.append(line_types_with_component)

        line_types = await prisma.linetype.find_many(
            where={"drawingTypeId": drawingType.id},
            order={"index": "asc"},
            include={"LineTypeComponent": {"include": {"Component": True}}},
        )
        if len(line_types) == 0:
            raise Exception("Line type not found")

        for line_type in line_types:
            print(line_type)

        # define: remaining components
        remaining_components_df = predicted_components_df.copy()

        # loop through line_types to diagnose the LineTypeComponent
        for line_type in line_types:
            # loop through the LineTypeComponents of the line type
            for line_type_component in line_type.LineTypeComponent:  # type: ignore
                # check if the component exists in the predicted components
                if (
                    line_type_component.Component.name  # type: ignore
                    in predicted_components_df["name"].values
                ):
                    # for each count
                    for i in range(line_type_component.count):
                        # get the index of the component
                        index = remaining_components_df[
                            remaining_components_df["name"]
                            == line_type_component.Component.name  # type: ignore
                        ].index[0]

                        # check if the count of the component is greater than 0
                        if remaining_components_df.loc[index, "count"] > 0:
                            # decrease the count of the component by 1
                            remaining_components_df.loc[index, "count"] -= 1

                        elif remaining_components_df.loc[index, "count"] == 0:
                            # remove the component from the dataframe
                            # remaining_components_df.drop(index, inplace=True)
                            continue
                        else:
                            raise Exception("Count of component is negative")

        print("Remaining components:")
        print(remaining_components_df)

        # close the database connection
        await prisma.disconnect()

        return remaining_components_df

    except Exception as e:
        print(e)
        return None


async def getIdComponents(drawing_components_df):
    try:
        print("Start getting id components")

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
                    drawing_components_df.loc[index, "id"] = component.id
                    break

        # close the database connection
        await prisma.disconnect()
        print("Finish getting id components")
        return drawing_components_df

    except Exception as e:
        print(e)

        return drawing_components_df


@app.route("/test-predict", methods=["POST"])
def test_predict():
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

        # get json result
        raw_json_result = results.pandas().xyxy[0].to_json(orient="records")
        string_json_result = asyncio.run(handle_json_result(raw_json_result))

        # create a df from the results
        df = results.pandas().xyxy[0]

        # count the number of components in each class
        predicted_components_df = df.groupby("name").size().reset_index(name="count")

        # validate the predicted components
        asyncio.run(validate_predicted_components(predicted_components_df))

        # diagnose the components
         remaining_components_df = asyncio.run(
            diagnose_components(predicted_components_df, drawing_type_id)
        )

        # add column key to drawing_components_df with value of row index
        drawing_components_df["key"] = drawing_components_df.index
        line_types_df["key"] = line_types_df.index
        missing_components_df["key"] = missing_components_df.index
        remaining_components_df["key"] = remaining_components_df.index

        # add column id to drawing_components_df with value of id from database
        drawing_components_df = asyncio.run(getIdComponents(drawing_components_df))

        # return all dfs to the client in json format
        line_types_json = line_types_df.to_json(orient="records")
        drawing_components_json = drawing_components_df.to_json(orient="records")
        missing_components_json = missing_components_df.to_json(orient="records")
        remaining_components_json = remaining_components_df.to_json(orient="records")

        # wait for the json result
        string_json_result = asyncio.run(string_json_result)

        response = make_response(
            {
                "line_types": line_types_json,
                "drawing_components": drawing_components_json,
                "missing_components": missing_components_json,
                "remaining_components": remaining_components_json,
                "json_result": string_json_result,
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


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
