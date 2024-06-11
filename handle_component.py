import os
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import RectangleSelector
from pydantic import BaseModel
from scipy.spatial import ConvexHull
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from skimage import morphology
from skimage.graph import route_through_array
from sklearn.cluster import AgglomerativeClustering

from prisma import Prisma


class DrawingComponentsDf(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    name: str
    color: str
    key: str
    lineTypeId: str
    lineTypeIdNumber: str
    lineTypeName: str
    group: int
    center_x: float
    center_y: float
    checked: bool
    cluster: int
    clusterLineTypeId: str


class HandleComponent:
    def __init__(
        self, debug: bool, df: pd.DataFrame, drawing_type_id: str, image_path: str
    ):
        try:
            self.debug = debug
            self.predicted_components_df = df.copy(deep=True).reset_index()
            self.drawing_type_id = drawing_type_id
            self.image_path = image_path
        except Exception as e:
            print(e)
            raise Exception(f"Error in HandleComponent: {e}")

    def get_predicted_components(self):
        return self.predicted_components_df

    async def get_detail_components(self):
        """
        Finds in the database componentId, color,
        and generates unique key
        then add them to each component in the dataframe
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

            # for each component in the self.predicted_components_df, check if all component names exist in components
            if not all(
                self.predicted_components_df["name"].isin(
                    [component.name for component in components]
                )
            ):
                raise Exception("Some components not found")

            for index, row in self.predicted_components_df.iterrows():
                for component in components:
                    if row["name"] == component.name:
                        self.predicted_components_df.at[index, "componentId"] = (
                            component.id
                        )
                        self.predicted_components_df.at[index, "color"] = (
                            component.color
                        )
                        self.predicted_components_df.at[index, "key"] = str(
                            uuid.uuid4()
                        )[:8]
                        break

            # get new color from nipy_spectral cmap in range 10% to 90%
            cmap = plt.cm.get_cmap("nipy_spectral", len(components))
            for index, row in self.predicted_components_df.iterrows():
                if row["color"] == None:
                    self.predicted_components_df.at[index, "color"] = colors.to_hex(
                        cmap(index / len(components))  # type: ignore
                    )

            # close the database connection
            await prisma.disconnect()
            return self.predicted_components_df

        except Exception as e:
            print(e)
            return None

        finally:
            print(f"---getIdComponents() {time.time() - time_start} seconds ---")

    async def fill_black_color(self, predicted_components_df: pd.DataFrame):
        """
        Fill black color on the image with the component bounding boxes
        and save the image to the output folder
        """
        try:
            # get the image
            image = mmcv.imread(self.image_path)

            # loop through the components
            for index, row in predicted_components_df.iterrows():
                # get the bounding box
                xmin, ymin, xmax, ymax = (
                    row["xmin"],
                    row["ymin"],
                    row["xmax"],
                    row["ymax"],
                )

                # make the bounding box integer
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                # fill the bounding box with white color
                image[ymin:ymax, xmin:xmax] = [255, 255, 255]

                # draw the bounding box with black color
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)

                # draw center axis of the bounding box
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                cv2.line(
                    image,
                    (center_x, ymin),
                    (center_x, ymax),
                    (0, 0, 0),
                    2,
                )
                cv2.line(
                    image,
                    (xmin, center_y),
                    (xmax, center_y),
                    (0, 0, 0),
                    2,
                )

            # save the image to the output folder
            output_folder = "images"
            filled_image_name = f"filled_{self.image_path.split('/')[-1]}"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # # display the image
            # mmcv.imshow(image)

            mmcv.imwrite(image, f"{output_folder}/{filled_image_name}")

            return

        except Exception as e:
            print(e)
            return None

    async def diagnose_components(self, image_path: str):
        """
        Order by index asc, for both line types and line type components.
        Loop through, check if the component exists in the predicted_components_df.
        If yes, then add it to the found components with additional information
        e.g. lineTypeId, group, center point, lineTypeIdNumber, checked.
        Then remove it from the remaining components.

        If no, add it to the missing components with additional information
        e.g. name, color, key, lineTypeId, lineTypeIdNumber, lineTypeName.
        """
        time_start = time.time()
        try:
            # database:
            prisma = Prisma()
            await prisma.connect()

            # query the database on drawing type table
            drawingType = await prisma.drawingtype.find_first(
                where={"id": self.drawing_type_id}
            )
            if drawingType == None:
                raise Exception("Drawing type not found")

            # get all line types with their components
            line_types = await prisma.linetype.find_many(
                where={"drawingTypeId": drawingType.id},
                order={"index": "asc"},
                include={"LineTypeComponent": {"include": {"Component": True}}},
            )
            if len(line_types) == 0:
                raise Exception("Line type not found")

            for line_type in line_types:
                if line_type.LineTypeComponent != None:
                    for line_type_component in line_type.LineTypeComponent:
                        if line_type_component.componentType == "mandatory":
                            line_type.LineTypeComponent.remove(line_type_component)
                            line_type.LineTypeComponent.insert(0, line_type_component)

            # define: remaining components
            remaining_components_df = self.predicted_components_df.copy()

            # define: missing components
            missing_components_df = self.predicted_components_df.copy().drop(
                self.predicted_components_df.index
            )

            # define: found components
            found_components_df = self.predicted_components_df.copy().drop(
                self.predicted_components_df.index
            )

            # get the highest y of 22_breaker
            index_highest_22_breaker = -1
            for o, row in remaining_components_df.iterrows():
                if row["name"] == "22_breaker":
                    if index_highest_22_breaker == -1:
                        index_highest_22_breaker = o
                    elif (
                        row["center_y"]
                        < remaining_components_df.loc[
                            index_highest_22_breaker, "center_y"
                        ]  # type: ignore
                    ):  # type: ignore
                        index_highest_22_breaker = o
            # get the highest y of 22_breaker
            highest_22_breaker_y = 5000
            if index_highest_22_breaker != -1:
                highest_22_breaker_y = remaining_components_df.loc[
                    index_highest_22_breaker, "center_y"
                ]  # type: ignore

            # loop through line_types to diagnose the LineTypeComponent
            for k in range(len(line_types)):
                line_type = line_types[k]

                for i in range(line_type.count):
                    # loop through the LineTypeComponents of the line type
                    mandatory_center_xy = [0, 0]

                    # print the length of dataframes
                    print(
                        f"line_type: {line_type.name}, remaining_components_df: {len(remaining_components_df)}, found_components_df: {len(found_components_df)}, missing_components_df: {len(missing_components_df)}"
                    )

                    for line_type_component in line_type.LineTypeComponent:  # type: ignore
                        for j in range(line_type_component.count):
                            if (
                                line_type_component.Component.name  # type: ignore
                                in remaining_components_df["name"].values
                            ):
                                # get the first index of the component in the remaining components
                                index = -1

                                # if the component is mandatory, save the center point
                                if (
                                    mandatory_center_xy == [0, 0]
                                    and line_type_component.componentType == "mandatory"
                                ):
                                    # if 115 sort by center_y, if 22 sort by center_x
                                    # sort the remaining components by the center_y
                                    if line_type.name.split("_")[0] == "115":
                                        # remaining_components_df = (
                                        #     remaining_components_df.sort_values(
                                        #         by=["center_x"], ascending=True
                                        #     )
                                        # ) ->
                                        remaining_components_df = (
                                            remaining_components_df.sort_values(
                                                by=["center_y"], ascending=True
                                            )
                                        )  # V
                                    elif line_type.name.split("_")[0] == "22":
                                        remaining_components_df = (
                                            remaining_components_df.sort_values(
                                                by=["center_x"], ascending=True
                                            )
                                        )  # ->

                                    index = remaining_components_df[
                                        remaining_components_df["name"]
                                        == line_type_component.Component.name  # type: ignore
                                    ].index[0]

                                    xmin, ymin, xmax, ymax = (
                                        remaining_components_df.loc[
                                            index, ["xmin", "ymin", "xmax", "ymax"]
                                        ]
                                    ).values

                                    mandatory_center_xy = [
                                        (xmin + xmax) / 2,
                                        (ymin + ymax) / 2,
                                    ]

                                # if the fisrt component is not mandatory, then find the closest component to the mandatory component
                                elif (
                                    mandatory_center_xy == [0, 0]
                                    and line_type_component.componentType == "optional"
                                ):
                                    # if 115 sort by center_y, if 22 sort by center_x
                                    # sort the remaining components by the center_y
                                    if line_type.name.split("_")[0] == "115":
                                        remaining_components_df = (
                                            remaining_components_df.sort_values(
                                                by=["center_y"], ascending=True
                                            )
                                        )
                                    elif line_type.name.split("_")[0] == "22":
                                        remaining_components_df = (
                                            remaining_components_df.sort_values(
                                                by=["center_x"], ascending=True
                                            )
                                        )

                                    index = remaining_components_df[
                                        remaining_components_df["name"]
                                        == line_type_component.Component.name  # type: ignore
                                    ].index[0]

                                    xmin, ymin, xmax, ymax = (
                                        remaining_components_df.loc[
                                            index, ["xmin", "ymin", "xmax", "ymax"]
                                        ]
                                    ).values

                                    mandatory_center_xy = [
                                        (xmin + xmax) / 2,
                                        (ymin + ymax) / 2,
                                    ]

                                    remaining_components_df.at[
                                        index, "componentType"
                                    ] = "mandatory"

                                else:
                                    # calculate the distance between the mandatory component and the remaining components
                                    remaining_components_df["distance"] = (
                                        (
                                            remaining_components_df["center_x"]
                                            - mandatory_center_xy[0]
                                        )
                                        ** 2
                                        + (
                                            remaining_components_df["center_y"]
                                            - mandatory_center_xy[1]
                                        )
                                        ** 2
                                    ) ** 0.5
                                    remaining_components_df["distance_x"] = (
                                        remaining_components_df["center_x"]
                                        - mandatory_center_xy[0]
                                    )
                                    remaining_components_df["distance_y"] = (
                                        remaining_components_df["center_y"]
                                        - mandatory_center_xy[1]
                                    )

                                    # get the busbar type
                                    busbar_type = line_type.name.split("_")[0]
                                    # if busbar_type == "115", then set distance to 0 for the components below the highest 22_breaker
                                    if busbar_type == "115":
                                        remaining_components_df.loc[
                                            remaining_components_df["center_y"]
                                            > highest_22_breaker_y,
                                            "distance",
                                        ] = 5000
                                    # if busbar_type == "22", then set distance to 0 for the components above the highest 22_breaker
                                    if busbar_type == "22":
                                        remaining_components_df.loc[
                                            remaining_components_df["center_y"]
                                            < highest_22_breaker_y,
                                            "distance",
                                        ] = 5000

                                    # get the closest component to the mandatory component with distance != 5000 with the same name
                                    index = -1
                                    for (
                                        c,
                                        component,
                                    ) in remaining_components_df.iterrows():
                                        if component["distance"] != 5000:
                                            if (
                                                component["name"]
                                                == line_type_component.Component.name  # type: ignore
                                            ):
                                                if (
                                                    index == -1
                                                    or component["distance"]
                                                    < remaining_components_df.loc[
                                                        index, "distance"
                                                    ]  # type: ignore
                                                ):
                                                    # if (
                                                    #     line_type.name == "22_incoming"
                                                    #     and component["distance_x"]
                                                    #     > 300
                                                    # ):
                                                    #     continue
                                                    # else:
                                                    #     index = c
                                                    index = c

                                    # plt.close()

                                    # if index == -1, add the component to the missing components
                                    if index == -1:
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
                                                            str(
                                                                uuid.uuid4()
                                                            )[  # generate a small uuid
                                                                :8
                                                            ]
                                                        ],
                                                        "lineTypeId": [
                                                            line_type_component.lineTypeId
                                                        ],
                                                        "lineTypeIdNumber": [
                                                            f"{line_type_component.lineTypeId}-{i}"
                                                        ],
                                                        "lineTypeName": [
                                                            f"{line_type.name}-{k}-{i}"
                                                        ],
                                                        "checked": [False],
                                                        "componentType": [
                                                            line_type_component.componentType
                                                        ],
                                                    }
                                                ),
                                            ],
                                            ignore_index=True,
                                        )

                                        continue

                                # if the component is not mandatory, then recalibrate the mandatory center point, with a weight factor
                                if line_type_component.componentType != "mandatory":
                                    # calculate the new center point with a weight factor
                                    newest_found_component = (
                                        remaining_components_df.loc[[index]]
                                    )
                                    weight_factor = 0.95
                                    mandatory_center_xy = [
                                        weight_factor * mandatory_center_xy[0]
                                        + (1 - weight_factor)
                                        * newest_found_component["center_x"].values[0],
                                        weight_factor * mandatory_center_xy[1]
                                        + (1 - weight_factor)
                                        * newest_found_component["center_y"].values[0],
                                    ]

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
                                found_components_df.at[last_index, "lineTypeId"] = (
                                    line_type_component.lineTypeId
                                )

                                # also add group number to the recently found component
                                found_components_df.at[last_index, "group"] = (
                                    f"{line_type_component.lineTypeId}-{i}"
                                )

                                # also add center point to the recently found component
                                found_components_df.at[last_index, "center_x"] = (
                                    found_components_df.at[last_index, "xmin"]
                                    + found_components_df.at[last_index, "xmax"]
                                ) / 2
                                found_components_df.at[last_index, "center_y"] = (
                                    found_components_df.at[last_index, "ymin"]
                                    + found_components_df.at[last_index, "ymax"]
                                ) / 2

                                # also add lineTypeIdNumber to the recently found component, which is a combination of lineTypeId and group
                                found_components_df.at[
                                    last_index, "lineTypeIdNumber"
                                ] = f"{line_type_component.lineTypeId}-{i}"

                                # add lineTypeName
                                found_components_df.at[last_index, "lineTypeName"] = (
                                    f"{line_type.name}-{k}-{i}"
                                )

                                # also add checked to the recently found component
                                found_components_df.at[last_index, "checked"] = False

                                # also add componentType to the recently found component
                                found_components_df.at[last_index, "componentType"] = (
                                    line_type_component.componentType
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
                                                "key": [
                                                    str(uuid.uuid4())[
                                                        :8
                                                    ]  # generate a small uuid
                                                ],
                                                "lineTypeId": [
                                                    line_type_component.lineTypeId
                                                ],
                                                "lineTypeIdNumber": [
                                                    f"{line_type_component.lineTypeId}-{i}"
                                                ],
                                                "lineTypeName": [
                                                    f"{line_type.name}-{k}-{i}"
                                                ],
                                                "checked": [False],
                                                "componentType": [
                                                    line_type_component.componentType
                                                ],
                                            }
                                        ),
                                    ],
                                    ignore_index=True,
                                )

                            # # # if line_type.name == "22_incoming":
                            # get unique line type names
                            # unique_line_type_names = found_components_df[
                            #     "lineTypeName"
                            # ].unique()
                            # # get unique colors
                            # unique_colors = plt.cm.get_cmap(
                            #     "nipy_spectral", len(unique_line_type_names)
                            # )
                            # # map the line type names to colors
                            # line_type_name_color = {}
                            # for n, line_type_name in enumerate(unique_line_type_names):
                            #     line_type_name_color[line_type_name] = colors.to_hex(
                            #         unique_colors(n)
                            #     )

                            # # plot the found components
                            # fig, ax = plt.subplots()
                            # image = mmcv.imread(image_path)
                            # ax.imshow(image)
                            # for k, component in found_components_df.iterrows():

                            #     rect = Rectangle(
                            #         (component["xmin"], component["ymin"]),
                            #         component["xmax"] - component["xmin"],
                            #         component["ymax"] - component["ymin"],
                            #         linewidth=1,
                            #         edgecolor=line_type_name_color[
                            #             component["lineTypeName"]
                            #         ],
                            #         facecolor="none",
                            #     )
                            #     ax.add_patch(rect)
                            #     ax.text(
                            #         component["xmin"],
                            #         component["ymin"],
                            #         f"{component['name']}-{component['lineTypeName']}",
                            #         fontsize=8,
                            #         color=line_type_name_color[
                            #             component["lineTypeName"]
                            #         ],
                            #     )

                            # # plot mandatory center point
                            # ax.scatter(
                            #     mandatory_center_xy[0],
                            #     mandatory_center_xy[1],
                            #     color="r",
                            #     s=100,
                            #     marker="x",
                            # )

                            # # save the image without showing
                            # plt.savefig("found_components.png")
                            # # plt.show()
                            # plt.close()

            # close the database connection
            await prisma.disconnect()

            # validate that found_components_df + remaining_components_df = predicted_components_df
            if len(self.predicted_components_df) != len(found_components_df) + len(
                remaining_components_df
            ):
                raise Exception("Error in diagnose: found + remaining != predicted")

            return found_components_df, remaining_components_df, missing_components_df

        except Exception as e:
            print(e)
            return None, None, None

        finally:
            print(f"---diagnose_components() {time.time() - time_start} seconds ---")

    def sort_line_type_components(self, found_components_df: pd.DataFrame):
        """
        Sorts the line type components by swapping the closest nodes
        1. pick one line type id
        2. pick one node(line type component) from the line type id that "checked"===false
        3. get the next node of that node
        4. get the closest node from all other line type ids inlcuding the current line type id
        5. mark the node "checked"=true, and if found, swap the next node with the closest node, go to step 2
        """

        try:
            # create a new dataframe to store the sorted line type components
            sorted_line_type_components_df = found_components_df.copy(deep=True)

            # get all line type ids
            line_type_ids = sorted_line_type_components_df["lineTypeId"].unique()

            # loop through all line type ids
            for line_type_id in line_type_ids:
                # get all line type components of the line type id
                line_type_components_df = sorted_line_type_components_df[
                    sorted_line_type_components_df["lineTypeId"] == line_type_id
                ]

                # get all unique groups of the line type id
                groups = line_type_components_df["group"].unique()

                # loop through all groups
                for group in groups:
                    # get all line type components of the group
                    group_df = line_type_components_df[
                        line_type_components_df["group"] == group
                    ]

                    # mark first node checked
                    sorted_line_type_components_df.at[
                        sorted_line_type_components_df[
                            sorted_line_type_components_df["key"]
                            == group_df.iloc[0]["key"]
                        ].index[0],
                        "checked",
                    ] = True

                    # get all line type components of the line type id
                    line_type_components_df = sorted_line_type_components_df[
                        sorted_line_type_components_df["lineTypeId"] == line_type_id
                    ]

                    # get all line type components of the group
                    group_df = line_type_components_df[
                        line_type_components_df["group"] == group
                    ]

                    # while there are still unchecked line type components in the group
                    while len(group_df[group_df["checked"] == False]) > 0:
                        # print the group
                        print(group_df)

                        # get the last checked node as pillar
                        pillars = group_df[group_df["checked"] == True]
                        if len(pillars) == 0:
                            continue
                        pillar = pillars.iloc[-1]

                        # get the next node that is not checked
                        next_nodes = group_df[group_df["checked"] == False]
                        if len(next_nodes) == 0:
                            continue
                        next_node = next_nodes.iloc[0]

                        # with the same node name, get the closest node from current and all other line type components
                        closest_nodes = sorted_line_type_components_df[
                            (
                                sorted_line_type_components_df["name"]
                                == next_node["name"]
                            )
                            & (sorted_line_type_components_df["checked"] == False)
                        ]

                        # if there is no closest node, then skip
                        if len(closest_nodes) == 0:
                            continue

                        # calculate the distance between the pillar and the closest node
                        closest_nodes["distance"] = (
                            (closest_nodes["center_x"] - pillar["center_x"]) ** 2
                            + (closest_nodes["center_y"] - pillar["center_y"]) ** 2
                        ) ** 0.5

                        # get the closest node to the pillar
                        closest_node = closest_nodes.iloc[
                            closest_nodes["distance"].argmin()
                        ]
                        if (
                            closest_node is None
                            or (closest_node["key"] == next_node["key"])
                            # or (
                            #     closest_node["lineTypeId"] == next_node["lineTypeId"]
                            #     and closest_node["group"] == next_node["group"]
                            # )
                        ):
                            # mark the next node checked
                            sorted_line_type_components_df.at[
                                sorted_line_type_components_df[
                                    sorted_line_type_components_df["key"]
                                    == next_node.key
                                ].index[0],
                                "checked",
                            ] = True

                            # get all line type components of the line type id
                            line_type_components_df = sorted_line_type_components_df[
                                sorted_line_type_components_df["lineTypeId"]
                                == line_type_id
                            ]
                            # get all line type components of the group
                            group_df = line_type_components_df[
                                line_type_components_df["group"] == group
                            ]

                            continue

                        swaps = [
                            "xmin",
                            "ymin",
                            "xmax",
                            "ymax",
                            "center_x",
                            "center_y",
                        ]
                        for swap in swaps:
                            old_next_node_value = sorted_line_type_components_df.loc[
                                sorted_line_type_components_df[
                                    sorted_line_type_components_df["key"]
                                    == next_node["key"]
                                ].index[0],
                                swap,
                            ]
                            old_closest_node_value = sorted_line_type_components_df.loc[
                                sorted_line_type_components_df[
                                    sorted_line_type_components_df["key"]
                                    == closest_node["key"]
                                ].index[0],
                                swap,
                            ]

                            sorted_line_type_components_df.loc[
                                sorted_line_type_components_df[
                                    sorted_line_type_components_df["key"]
                                    == next_node["key"]
                                ].index[0],
                                swap,
                            ] = old_closest_node_value
                            sorted_line_type_components_df.loc[
                                sorted_line_type_components_df[
                                    sorted_line_type_components_df["key"]
                                    == closest_node["key"]
                                ].index[0],
                                swap,
                            ] = old_next_node_value

                        # mark the next node checked
                        sorted_line_type_components_df.at[
                            sorted_line_type_components_df[
                                sorted_line_type_components_df["key"] == next_node.key
                            ].index[0],
                            "checked",
                        ] = True

                        # get all line type components of the line type id
                        line_type_components_df = sorted_line_type_components_df[
                            sorted_line_type_components_df["lineTypeId"] == line_type_id
                        ]
                        # get all line type components of the group
                        group_df = line_type_components_df[
                            line_type_components_df["group"] == group
                        ]

                        continue

            return sorted_line_type_components_df

        except Exception as e:
            print(e)
            return None

    async def get_clustered_components(
        self, found_components_df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Clusters the found components into groups
        """
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

            # get line types of this drawingType
            # database:
            prisma = Prisma()
            await prisma.connect()
            lineTypes = await prisma.linetype.find_many(
                where={"drawingTypeId": self.drawing_type_id}
            )

            # n_clusters is the number of line types * its count
            n_clusters = sum([lineType.count for lineType in lineTypes]) / 2
            # parse into int
            n_clusters = int(n_clusters)

            # Perform clustering
            clustering = AgglomerativeClustering(
                n_clusters, affinity="precomputed", linkage="average"
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

            # close the database connection
            await prisma.disconnect()

            return clustered_found_components_df

        except Exception as e:
            print(e)
            return None

    def create_convexhull(
        self,
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

    async def get_clustered_convexhull(
        self,
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
            hulls = pd.DataFrame(
                columns=["clusterLineTypeId", "points", "key", "clusterLineTypeName"]
            )

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
                hull = self.create_convexhull(line_type_components)

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
                                "clusterLineTypeId": [row["clusterLineTypeId"]],
                                "points": [hull[0]],
                                "key": [row["key"]],
                                "clusterLineTypeName": [line_type.name],
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

    async def get_found_convexhull(
        self,
        found_components_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Returns the convex hull of the found components
        """
        time_start = time.time()
        try:
            # group by line type id & group and generate new uuid for each group
            groups = pd.DataFrame(
                found_components_df.groupby(["lineTypeIdNumber"]).size().reset_index()
            )
            groups["key"] = [str(uuid.uuid4())[:8] for i in range(len(groups))]

            # create a new dataframe to store the hull of each line type id
            hulls = pd.DataFrame(
                columns=["foundLineTypeId", "points", "key", "foundLineTypeName"]
            )

            for index, row in groups.iterrows():
                # get the line type components of the line type id
                line_type_components = found_components_df[
                    found_components_df["lineTypeIdNumber"] == row["lineTypeIdNumber"]
                ]

                # skip for line type components length < 3
                if len(line_type_components) < 3:
                    continue

                # get the convex hull of the line type components
                hull = self.create_convexhull(line_type_components)

                # get the line type name from db
                prisma = Prisma()
                await prisma.connect()
                line_type_id = row["lineTypeIdNumber"].split("-")[0]
                line_type = await prisma.linetype.find_first(where={"id": line_type_id})
                if line_type == None:
                    raise Exception("Line type not found")

                hulls = pd.concat(
                    [
                        hulls,
                        pd.DataFrame(
                            {
                                "foundLineTypeId": [row["lineTypeIdNumber"]],
                                "points": [hull[0]],
                                "key": [row["key"]],
                                "foundLineTypeName": [line_type.name],
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

    async def correct_missing_component(
        self,
        clustered_found_components_df: pd.DataFrame,
        missing_components_df: pd.DataFrame,
        clustered_hulls: pd.DataFrame,
    ):
        """
        For each component in clustered_found_components_df
            If a component is not in its clusterConvexHull
                If the component is in missing_component_df
                    Then correct it (change line type id, line type id number, line type name, checked)
        """
        time_start = time.time()
        try:
            for (
                i,
                clustered_found_component,
            ) in clustered_found_components_df.iterrows():
                # get found component cluster id
                found_component_cluster_id = clustered_found_component[
                    "clusterLineTypeId"
                ]
                # get found component cluster convex hull
                cluster_hull = clustered_hulls[
                    clustered_hulls["clusterLineTypeId"] == found_component_cluster_id
                ]

                # if the cluster convex hull is empty, then skip
                if len(cluster_hull) == 0:
                    continue

                # if the component is in the cluster convex hull, then skip
                if any(
                    [
                        clustered_found_component["center_x"] == point["x"]
                        and clustered_found_component["center_y"] == point["y"]
                        for point in cluster_hull["points"].values[0]
                    ]
                ):
                    continue

                print(f"found_component: {clustered_found_component}")

                clustered_found_component_name = clustered_found_component["name"]

                # if the component name is in the missing components that checked==False, then correct it
                if len(missing_components_df) > 0 and (
                    clustered_found_component_name
                    in missing_components_df[
                        (missing_components_df["checked"] == False)
                    ]["name"].values
                ):
                    print(
                        f"missing_component: {missing_components_df[missing_components_df['name'] == clustered_found_component_name]}"
                    )
                    # change missing component line type id to found component line type id
                    missing_components_df.at[
                        missing_components_df[
                            missing_components_df["name"]
                            == clustered_found_component_name
                        ].index[0],
                        "lineTypeId",
                    ] = clustered_found_component["lineTypeId"]
                    # change missing component line type id number to found component line type id number
                    missing_components_df.at[
                        missing_components_df[
                            missing_components_df["name"]
                            == clustered_found_component_name
                        ].index[0],
                        "lineTypeIdNumber",
                    ] = clustered_found_component["lineTypeIdNumber"]
                    # change missing component line type name to found component line type name
                    missing_components_df.at[
                        missing_components_df[
                            missing_components_df["name"]
                            == clustered_found_component_name
                        ].index[0],
                        "lineTypeName",
                    ] = clustered_found_component["lineTypeName"]
                    # checked to true
                    missing_components_df.at[
                        missing_components_df[
                            missing_components_df["name"]
                            == clustered_found_component_name
                        ].index[0],
                        "checked",
                    ] = True

            return missing_components_df

        except Exception as e:
            print(e)
            raise e

        finally:
            print(f"---getIdComponents() {time.time() - time_start} seconds ---")
