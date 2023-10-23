import os
import time
import uuid

import pandas as pd

from prisma import Prisma


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

    async def get_detail_components(self):
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

            # # for each component in the self.predicted_components_df, check if all component names exist in components
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

    async def diagnose_components(self):
        # predicted_components_df: pd.DataFrame, drawing_type_id: str
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

            # loop through line_types to diagnose the LineTypeComponent
            for line_type in line_types:
                for i in range(line_type.count):
                    # loop through the LineTypeComponents of the line type
                    for line_type_component in line_type.LineTypeComponent:  # type: ignore
                        for j in range(line_type_component.count):
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

                                # also add group number to the recently found component
                                found_components_df.at[last_index, "group"] = i

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

                                # also add checked to the recently found component
                                found_components_df.at[last_index, "checked"] = False

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
                                                "lineTypeName": [line_type.name],
                                            }
                                        ),
                                    ],
                                    ignore_index=True,
                                )

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
        # algorithm to swap closest points to correct the group
        # 1. pick one line type id
        # 2. pick one node from the line type id that "checked"===false
        # 3. get the next node from the line type id
        # 4. get the closest node from all other line type ids inlcuding the current line type id
        # 5. mark the node "checked"=true, and if found, swap the next node with the closest node, go to step 2

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

                        # swap the next node with the closest node
                        # xmin, ymin, xmax, ymax, center_x, center_y, lineTypeId, group
                        # print(
                        #     "old_next_node",
                        #     sorted_line_type_components_df.at[next_node.key],
                        # )
                        # print(
                        #     "old_closest_node",
                        #     sorted_line_type_components_df.at[closest_node.key],
                        # )

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

                        # print(
                        #     "new_next_node",
                        #     sorted_line_type_components_df.at[next_node.key],
                        # )
                        # print(
                        #     "new_closest_node",
                        #     sorted_line_type_components_df.at[closest_node.key],
                        # )

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
