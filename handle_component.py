import os
import time
import uuid

import pandas as pd

from prisma import Prisma


class HandleComponent:
    def __init__(self, df: pd.DataFrame):
        if df.empty or df is None:
            raise Exception("Empty dataframe")
        self.predicted_components_df = df.copy(deep=True).reset_index()

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

        # finally:
        #     print(f"---getIdComponents() {time.time() - time_start} seconds ---")
