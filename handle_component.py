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

    async def getDetailComponents(self):
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
