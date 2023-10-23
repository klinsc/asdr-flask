import asyncio
import json
import os
import sys
import time
import uuid
from collections import Counter
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, make_response, request
from flask_cors import CORS
from pdf2image.pdf2image import convert_from_bytes
from PIL import Image
from scipy.spatial import ConvexHull
from sklearn.cluster import AgglomerativeClustering

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


def getClusteredComponents(found_components_df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        clustered_found_components_df = found_components_df.copy(deep=True)

        # Create a dataset from the list of found components
        posX = []
        posY = []
        for index, row in clustered_found_components_df.iterrows():
            posX.append(row["center_x"])
            posY.append(row["center_y"])
        nodes = np.array([posX, posY]).T
        # flip upside down
        nodes[:, 1] = -nodes[:, 1]

        # Define a custom distance metric
        def max_distance(node1, node2):
            return max(abs(node1[0] - node2[0]), abs(node1[1] - node2[1]))

        # Create a distance matrix
        distance_matrix = np.array(
            [[max_distance(node1, node2) for node2 in nodes] for node1 in nodes]
        )

        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=10, affinity="precomputed", linkage="average"
        )
        clustering.fit(distance_matrix)

        # Get line type ids as metadata for each node
        metadata = clustered_found_components_df["lineTypeIdNumber"].values

        # Create a dictionary that maps each node to its metadata
        node_to_metadata = {
            tuple(node): line_type_id for node, line_type_id in zip(nodes, metadata)
        }

        # Create a mapping from cluster labels to metadata categories
        cluster_to_metadata = {}

        for label in set(clustering.labels_):
            # Get the metadata categories for all nodes in this cluster
            category_in_cluster = [
                node_to_metadata[tuple(node)]
                for node in nodes[clustering.labels_ == label]
            ]

            # Find the most common category
            most_common_category = Counter(category_in_cluster).most_common(1)[0][0]

            # Map the cluster label to the most common category
            cluster_to_metadata[label] = most_common_category

        # add the cluster to the dataframe with normalised cluster labels (start from 0,1)
        clustered_found_components_df["cluster"] = clustering.labels_

        # add the cluster metadata to the dataframe
        clustered_found_components_df["clusterLineTypeId"] = [
            cluster_to_metadata[cluster]
            for cluster in clustered_found_components_df["cluster"].values
        ]

        # # add the cluster to the dataframe
        # clustered_found_components_df["cluster"] = clustering.labels_

        # /10 to normalise the cluster labels
        clustered_found_components_df["cluster"] = (
            clustered_found_components_df["cluster"] / 10
        )

        return clustered_found_components_df

    except Exception as e:
        print(e)
        return None


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
        print(found_components_df)
        # group by line type id & group and generate new uuid for each group
        groups = pd.DataFrame(
            found_components_df.groupby(["lineTypeId", "group"]).size()
        ).reset_index()
        groups["key"] = [str(uuid.uuid4())[:8] for i in range(len(groups))]
        print(groups)

        # create a new dataframe to store the hull of each line type id
        hulls = pd.DataFrame(columns=["lineTypeId", "points", "key"])

        for index, row in groups.iterrows():
            # get the line type components of the line type id
            line_type_components = found_components_df[
                (found_components_df["lineTypeId"] == row["lineTypeId"])
                & (found_components_df["group"] == row["group"])
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
                            "points": [hull[0]],
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


async def getClusteredConvexHull(
    clustered_found_components_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns the convex hull of the clustered found components
    """
    time_start = time.time()
    try:
        # group by line type id & group and generate new uuid for each group
        groups = pd.DataFrame(
            clustered_found_components_df.groupby(["clusterLineTypeId", "cluster"])
            .size()
            .reset_index()
        )
        groups["key"] = [str(uuid.uuid4())[:8] for i in range(len(groups))]

        # create a new dataframe to store the hull of each line type id
        hulls = pd.DataFrame(columns=["lineTypeId", "points", "key", "lineTypeName"])

        for index, row in groups.iterrows():
            # get the line type components of the line type id
            line_type_components = clustered_found_components_df[
                (
                    clustered_found_components_df["clusterLineTypeId"]
                    == row["clusterLineTypeId"]
                )
                & (clustered_found_components_df["cluster"] == row["cluster"])
            ]

            # skip for line type components length < 3
            if len(line_type_components) < 3:
                continue

            # get the convex hull of the line type components
            hull = getLineTypeConvexHull(line_type_components)

            # get the line type name from db
            prisma = Prisma()
            await prisma.connect()
            line_type_id = row["clusterLineTypeId"].split("-")[0]
            line_type = await prisma.linetype.find_first(where={"id": line_type_id})
            if line_type == None:
                raise Exception("Line type not found")

            # add the hull to the hulls dataframe
            hulls = pd.concat(
                [
                    hulls,
                    pd.DataFrame(
                        {
                            "lineTypeId": [row["clusterLineTypeId"]],
                            "points": [hull[0]],
                            "key": [row["key"]],
                            "lineTypeName": [line_type.name],
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
        new_found_components_df = component_handler.sort_line_type_components(
            found_components_df
        )
        if new_found_components_df is None:
            raise Exception("Error in sort line type components")

        # validate that found_components_df + remaining_components_df = predicted_components_df
        if len(predicted_components_df) != len(new_found_components_df) + len(
            remaining_components_df
        ):
            raise Exception("Error in sort: found + remaining != predicted")

        # # get hulls
        # hulls = getFoundComponentsConvexHull(new_found_components_df)
        # hulls = hulls.to_json(orient="records")

        clustered_found_components_df = getClusteredComponents(new_found_components_df)
        if clustered_found_components_df is None:
            raise Exception("Error in cluster components")

            # get hulls
        hulls = asyncio.run(getClusteredConvexHull(clustered_found_components_df))
        hulls = hulls.to_json(orient="records")

        # return all dfs to the client in json format
        predicted_components_json = predicted_components_df.to_json(orient="records")
        found_components_json = found_components_df.to_json(orient="records")
        missing_components_json = missing_components_df.to_json(orient="records")
        remaining_components_json = remaining_components_df.to_json(orient="records")
        clustered_found_components_json = clustered_found_components_df.to_json(
            orient="records"
        )

        response = make_response(
            {
                "predicted_components": predicted_components_json,
                "found_components": found_components_json,
                "remaining_components": remaining_components_json,
                "missing_components": missing_components_json,
                "hulls": hulls,
                "clustered_found_components": clustered_found_components_json,
            },
            200,
            {"Content-Type": "application/json"},
        )

        # # temp
        # response = make_response(string_json_result, 200, {"Content-Type": "json"})

        return response
    except Exception as e:
        print(e)

        return make_response(f"Internal Server Error: {e}", 500)
    finally:
        print(f"---test_predict() {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
