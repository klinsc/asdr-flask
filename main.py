import asyncio
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

# create a flask server to receive the pdf file and convert it to images and send it back to the client
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def index():
    return "asdr-flask-server"


@app.route("/health")
def health():
    return "OK"


@app.route("/upload", methods=["POST"])
def upload():
    try:
        print(request)
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
                component = await prisma.component.find_unique(
                    where={"name": bounding_box.name}
                )
                if component == None:
                    continue

                # add the color to the dataframe
                newResults.at[bounding_box.Index, "color"] = component.color

            # close the database connection
            await prisma.disconnect()

            # for each bounding box, generate a small uuid for each
            newResults["id"] = newResults.apply(
                lambda row: str(uuid.uuid4())[:8], axis=1
            )

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


async def validate_bays(drawing_components_df, drawing_type="Main&Transfer", test=True):
    """validate the drawing components
    Args:
        drawing_components_df (pd.DataFrame): dataframe of drawing components with columns: name, count
        drawing_type (str, optional): drawing type. Defaults to "Main&Transfer".

    Returns:
    """
    try:
        print("Start validating bay")

        # database:
        prisma = Prisma()
        await prisma.connect()

        # args:
        if test:
            # queries the drawing table on the database
            drawing = await prisma.drawing.find_unique(
                where={"name": "e4a96435-bwa-BangWua1-sm-mt"}
            )
            if drawing == None:
                raise Exception("Drawing not found")

            # queries all classes in the database on component table
            drawing_components = await prisma.drawingcomponent.find_many(
                where={"drawingId": drawing.id}, include={"component": True}
            )
            if len(drawing_components) == 0:
                raise Exception("Component not found")

            # create a list of components
            drawing_components_list = []
            for drawing_component in drawing_components:
                if drawing_component.component == None:
                    continue

                drawing_components_list.append(
                    {
                        "index": drawing_component.component.index,
                        "name": drawing_component.component.name,
                        "count": drawing_component.count,
                    }
                )
            # convert to dataframe
            drawing_components_df = pd.DataFrame(drawing_components_list)

        print("drawing_components_df:", drawing_components_df)
        drawing_components_df_backup = drawing_components_df.copy()

        # query the database on drawing type table
        drawingType = await prisma.drawingtype.find_first(where={"name": drawing_type})
        if drawingType == None:
            raise Exception("Drawing type not found")
        # queries all lines in the database on line type table
        drawing_line_types = await prisma.linetype.find_many(
            where={"drawingTypeId": drawingType.id}
        )
        if len(drawing_line_types) == 0:
            raise Exception("Line not found")

        # create a list of line types with count = 0
        line_types = []
        for line_type in drawing_line_types:
            line_types.append({"name": line_type.name, "count": 0})
        # convert to dataframe
        line_types_df = pd.DataFrame(line_types)

        # !!! count lines in the drawing !!!
        # 1) count "115_tie" in line_types (default is 1),
        line_types_df.loc[line_types_df["name"] == "115_tie", "count"] += 1
        # and for "115_vt_1p" or "115_cvt_1p" or "115_vt_3p" or "115_cvt_3p" which find first remove by 1 from drawing_components_list
        for index, row in drawing_components_df.loc[
            (drawing_components_df["name"] == "115_vt_1p")
            | (drawing_components_df["name"] == "115_cvt_1p")
            | (drawing_components_df["name"] == "115_vt_3p")
            | (drawing_components_df["name"] == "115_cvt_3p")
        ].iterrows():
            if row["count"]:
                drawing_components_df.loc[index, "count"] -= 1  # type: ignore
                break

        # 2) count "115_transformer" in line_types by counting "11522_tx_dyn1" or "11522_tx_ynyn0d1" in drawing_components_list
        line_types_df.loc[
            line_types_df["name"] == "115_transformer", "count"
        ] += drawing_components_df.loc[
            (drawing_components_df["name"] == "11522_tx_dyn1")
            | (drawing_components_df["name"] == "11522_tx_ynyn0d1"),
            "count",
        ].sum()

        # 3) count "115_incoming" in line_types by counting "115_vt_1p" or "115_cvt_1p" or "115_vt_3p" or "115_cvt_3p"
        line_types_df.loc[
            line_types_df["name"] == "115_incoming", "count"
        ] += drawing_components_df.loc[
            (drawing_components_df["name"] == "115_vt_1p")
            | (drawing_components_df["name"] == "115_cvt_1p")
            | (drawing_components_df["name"] == "115_vt_3p")
            | (drawing_components_df["name"] == "115_cvt_3p"),
            "count",
        ].sum()

        # 4) count "22_tie" in line_types (default is 1)
        line_types_df.loc[line_types_df["name"] == "22_tie", "count"] += 1

        # 5) count "22_capacitor" in line_types by counting "22_cap_bank" in drawing_components_list
        line_types_df.loc[
            line_types_df["name"] == "22_capacitor", "count"
        ] += drawing_components_df.loc[
            (drawing_components_df["name"] == "22_cap_bank"), "count"
        ].sum()

        # 6) count "22_outgoing" in line_types by counting the "22_ds_la_out" or "22_ds_out" in drawing_components_list
        line_types_df.loc[
            line_types_df["name"] == "22_outgoing", "count"
        ] += drawing_components_df.loc[
            (drawing_components_df["name"] == "22_ds_la_out")
            | (drawing_components_df["name"] == "22_ds_out"),
            "count",
        ].sum()

        # 7) count "22_incoming" in line_types by counting the "v_m" in drawing_components_list
        line_types_df.loc[
            line_types_df["name"] == "22_incoming", "count"
        ] += drawing_components_df.loc[
            (drawing_components_df["name"] == "v_m"), "count"
        ].sum()

        # 8) count "22_service" in line_types by counting the "22_ds" in drawing_components_list
        line_types_df.loc[
            line_types_df["name"] == "22_service", "count"
        ] += drawing_components_df.loc[
            (drawing_components_df["name"] == "22_ds"), "count"
        ].sum()

        # 9) reset drawing_components_df to the original dataframe
        drawing_components_df = drawing_components_df_backup.copy()

        # # 10) remove the components with count = 0
        # drawing_components_df = drawing_components_df[drawing_components_df["count"]
        #                                               != 0].reset_index(drop=True)

        # # 99) conclue to the number of total 115_incoming, 115_transformer, 115_tie
        print("line_types_df:", line_types_df)

        await prisma.disconnect()

        print("Finish validating bays")
        return line_types_df, drawing_components_df

    except Exception as e:
        print(e)

        return None, None


async def validate_components(
    drawing_components_df, line_types_df, drawing_type="Main&Transfer"
):
    try:
        print("Start validating components")

        drawing_truth = drawing_tree[drawing_type]
        missing_components_df = pd.DataFrame(columns=["name", "line_type", "count"])
        remaining_components_df = drawing_components_df.copy()

        # for each line type in 115 of line_types_df
        for index, row in line_types_df.iterrows():
            # get the line type name
            line_type_name = row["name"]
            # get the line type count
            line_type_count = row["count"]
            # get mandatory components
            mandatories = drawing_truth[line_type_name]["mandatory"]

            for i in range(line_type_count):
                # for each mandatory component in the line type
                for mandatory in mandatories:
                    # check if the mandatory component has no variant
                    if isinstance(mandatories[mandatory], int):
                        # get the mandatory component count
                        mandatory_count = mandatories[mandatory]

                        while mandatory_count > 0:
                            founded = False

                            # if there is a mandatory component in the drawing_components_df (not None)
                            if remaining_components_df.loc[
                                remaining_components_df["name"] == mandatory, "count"
                            ].any():
                                # deduct the mandatory component count from the drawing_components_df
                                remaining_components_df.loc[
                                    remaining_components_df["name"] == mandatory,
                                    "count",
                                ] -= 1
                                # deduct the mandatory component count
                                mandatory_count -= 1

                                founded = True

                            if founded == False:
                                # add missing mandatory component to missing_components_df
                                missing_components_df = missing_components_df._append(
                                    {
                                        "name": mandatory,
                                        "line_type": line_type_name,
                                        "count": 1,
                                    },
                                    ignore_index=True,
                                )  # type: ignore

                                # deduct the mandatory component count
                                mandatory_count -= 1

                    # means the mandatory component has variants
                    else:
                        # get the mandatory component variants
                        mandatory_component_variants = mandatories[mandatory]

                        # get its _total truth
                        mandatory_component_variants_total = (
                            mandatory_component_variants["_total"]
                        )

                        while mandatory_component_variants_total > 0:
                            founded = False

                            for variant in mandatory_component_variants:
                                if variant == "_total":
                                    continue

                                # if there is a variant in the drawing_components_df (not None)
                                if remaining_components_df.loc[
                                    remaining_components_df["name"] == variant, "count"
                                ].any():
                                    # deduct the variant count from the drawing_components_df
                                    remaining_components_df.loc[
                                        remaining_components_df["name"] == variant,
                                        "count",
                                    ] -= 1
                                    # deduct the variant count from the mandatory_component_variants_total
                                    mandatory_component_variants_total -= 1

                                    founded = True
                                    break

                            if founded == False:
                                # add missing variant to missing_components_df
                                missing_components_df = missing_components_df._append(
                                    {
                                        "name": mandatory,
                                        "line_type": line_type_name,
                                        "count": 1,
                                    },
                                    ignore_index=True,
                                )  # type: ignore

                                # deduct the variant count from the mandatory_component_variants_total
                                mandatory_component_variants_total -= 1

        print("Remaining drawing_components_df:", remaining_components_df)
        print("Missing components:", missing_components_df)
        print("Finish validating components")

        return missing_components_df, remaining_components_df

    except Exception as e:
        print(e)
        return None, None


@app.route("/test-predict", methods=["POST"])
def test_predict():
    test = True if (request.args.get("test") == "true") else False

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

        # create a df from the results
        df = results.pandas().xyxy[0]

        # count the number of components in each class
        df = df.groupby("name").size().reset_index(name="count")

        # validate the drawing bays
        line_types_df, drawing_components_df = asyncio.run(validate_bays(df, test=test))

        # validate the components
        missing_components_df, remaining_components_df = asyncio.run(
            validate_components(drawing_components_df, line_types_df)
        )

        # add column key to drawing_components_df with value of row index
        drawing_components_df["key"] = drawing_components_df.index
        line_types_df["key"] = line_types_df.index
        missing_components_df["key"] = missing_components_df.index
        remaining_components_df["key"] = remaining_components_df.index

        # return all dfs to the client in json format
        line_types_json = line_types_df.to_json(orient="records")
        drawing_components_json = drawing_components_df.to_json(orient="records")
        missing_components_json = missing_components_df.to_json(orient="records")
        remaining_components_json = remaining_components_df.to_json(orient="records")

        response = make_response(
            {
                "line_types": line_types_json,
                "drawing_components": drawing_components_json,
                "missing_components": missing_components_json,
                "remaining_components": remaining_components_json,
            },
            200,
            {"Content-Type": "application/json"},
        )

        return response
    except Exception as e:
        print(e)

        return make_response("Internal Server Error", 500)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
