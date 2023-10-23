import os
import time
import uuid
from collections import Counter

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.spatial import ConvexHull
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
    def __init__(self, df: pd.DataFrame, drawing_type_id: str):
        try:
            self.predicted_components_df = df.copy(deep=True).reset_index()
            self.drawing_type_id = drawing_type_id
        except Exception as e:
            print(e)
            raise Exception(f"Error in HandleComponent: {e}")

    def get_predicted_components(self):
        return self.predicted_components_df

    async def get_n_clusters(self):
        # database:
        prisma = Prisma()
        await prisma.connect()

        # get all line types with their count in the database from the drawing type id
        line_types = await prisma.linetype.find_many(
            where={"drawingTypeId": self.drawing_type_id},
            order={"index": "asc"},
        )

        n_clusters = 0
        for line_type in line_types:
            n_clusters += line_type.count

        # close the database connection
        await prisma.disconnect()

        return n_clusters

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
                        self.predicted_components_df.at[
                            index, "componentId"
                        ] = component.id
                        self.predicted_components_df.at[
                            index, "color"
                        ] = component.color
                        self.predicted_components_df.at[index, "key"] = str(
                            uuid.uuid4()
                        )[:8]
                        break

            # close the database connection
            await prisma.disconnect()
            return self.predicted_components_df

        except Exception as e:
            print(e)
            return None

        finally:
            print(f"---getIdComponents() {time.time() - time_start} seconds ---")

    async def diagnose_components(
        self, clustered_predicted_components_df: pd.DataFrame
    ):
        """ """
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

            # define line type components dataframe
            line_type_components_df = pd.DataFrame(
                columns=[
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                    "name",
                    "color",
                    "componentId",
                    "lineTypeId",
                    "lineTypeIdNumber",
                    "lineTypeName",
                    "cluster",
                    "center_x",
                    "center_y",
                    "checked",
                ]
            )

            # loop through all line types
            for line_type in line_types:
                for i in range(line_type.count):
                    for line_type_component in line_type.LineTypeComponent:  # type: ignore
                        for j in range(line_type_component.count):
                            # add the line type component to the dataframe
                            line_type_components_df = pd.concat(
                                [
                                    line_type_components_df,
                                    pd.DataFrame(
                                        {
                                            "xmin": [None],
                                            "ymin": [None],
                                            "xmax": [None],
                                            "ymax": [None],
                                            "name": [line_type_component.Component.name],  # type: ignore
                                            "color": [line_type_component.Component.color],  # type: ignore
                                            "componentId": [line_type_component.Component.id],  # type: ignore
                                            "lineTypeId": [line_type.id],
                                            "lineTypeIdNumber": [f"{line_type.id}-{i}"],
                                            "lineTypeName": [line_type.name],
                                            "cluster": [None],
                                            "center_x": [None],
                                            "center_y": [None],
                                            "checked": [False],
                                        }
                                    ),
                                ],
                                ignore_index=True,
                            )

            # define: remaining components
            remaining_components_df = clustered_predicted_components_df.copy()

            # define: missing components
            missing_components_df = clustered_predicted_components_df.copy().drop(
                clustered_predicted_components_df.index
            )

            # define: found components
            found_components_df = clustered_predicted_components_df.copy().drop(
                clustered_predicted_components_df.index
            )

            # match each clustered_predicted_components_df with each line_type_components_df
            # by comparing the name, color, componentId and number of components
            for index, line_type_component in line_type_components_df.iterrows():
                # get all components with the lineTypeIdNumber in line_type_components_df
                line_type_components = line_type_components_df[
                    line_type_components_df["lineTypeIdNumber"]
                    == line_type_component["lineTypeIdNumber"]
                ]

            # close the database connection
            await prisma.disconnect()

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

    async def get_clustered_components(self) -> pd.DataFrame | None:
        try:
            clustered_found_components_df = self.predicted_components_df.copy(deep=True)

            # create center_x & center_y columns
            clustered_found_components_df["center_x"] = (
                clustered_found_components_df["xmin"]
                + clustered_found_components_df["xmax"]
            ) / 2
            clustered_found_components_df["center_y"] = (
                clustered_found_components_df["ymin"]
                + clustered_found_components_df["ymax"]
            ) / 2

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

            n_clusters = await self.get_n_clusters()

            # Perform clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                linkage="average",
            )
            clustering.fit(distance_matrix)

            # add the cluster to the dataframe with normalised cluster labels (start from 0,1)
            clustered_found_components_df["cluster"] = clustering.labels_

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
                columns=["lineTypeId", "points", "key", "lineTypeName"]
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
