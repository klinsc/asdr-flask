import asyncio
from prisma import Prisma
import pandas as pd
from drawingTree import drawingTree


async def countBay(drawing_name: str):
    try:
        print("Start counting bay")

        prisma = Prisma()
        await prisma.connect()

        # queries the drawing table on the database
        drawing = await prisma.drawing.find_unique(
            where={
                "name": drawing_name
            }
        )
        if (drawing == None):
            raise Exception("Drawing not found")

        # queries all classes in the database on component table
        drawing_components = await prisma.drawingcomponent.find_many(
            where={
                "drawingId": drawing.id
            },
            include={
                "component": True
            }
        )
        if (len(drawing_components) == 0):
            raise Exception("Component not found")

        # create a list of components
        drawing_components_list = []
        for drawing_component in drawing_components:
            drawing_components_list.append({
                "index": drawing_component.component.index,
                "name": drawing_component.component.name,
                "count": drawing_component.count
            })
        # convert to dataframe
        drawing_components_df = pd.DataFrame(drawing_components_list)
        drawing_components_df_backup = drawing_components_df.copy()

        # queries all lines in the database on line type table
        drawing_line_types = await prisma.linetype.find_many(
            where={
                "drawingTypeId": drawing.drawingTypeId
            }
        )
        if (len(drawing_line_types) == 0):
            raise Exception("Line not found")

        # create a list of line types with count = 0
        line_types = []
        for line_type in drawing_line_types:
            line_types.append({
                "name": line_type.name,
                "count": 0
            })
        # convert to dataframe
        line_types_df = pd.DataFrame(line_types)

        # !!! count lines in the drawing !!!
        # 1) count "115_tie" in line_types (default is 1) and minus 1 from the count of "115_breaker" in drawing_components_list
        line_types_df.loc[line_types_df["name"] == "115_tie", "count"] += 1
        drawing_components_df.loc[drawing_components_df["name"]
                                  == "115_breaker", "count"] -= 1

        # 2) count "115_transformer" in line_types by counting "11522_tx_dyn1" or "11522_tx_ynyn0d1" in drawing_components_list,
        line_types_df.loc[line_types_df["name"] == "115_transformer", "count"] += drawing_components_df.loc[(
            drawing_components_df["name"] == "11522_tx_dyn1") | (drawing_components_df["name"] == "11522_tx_ynyn0d1"), "count"].sum()
        # and minus number of "115_breaker" in drawing_components_list with the number of "115_transformer" in line_types
        drawing_components_df.loc[drawing_components_df["name"] == "115_breaker",
                                  "count"] -= line_types_df.loc[line_types_df["name"] == "115_transformer", "count"].sum()
        # and minus number of "11522_tx_dyn1"
        drawing_components_df.loc[drawing_components_df["name"] == "11522_tx_dyn1",
                                  "count"] -= line_types_df.loc[line_types_df["name"] == "115_transformer", "count"].sum()
        # or "11522_tx_ynyn0d1" in drawing_components_list
        drawing_components_df.loc[drawing_components_df["name"] == "11522_tx_ynyn0d1",
                                  "count"] -= line_types_df.loc[line_types_df["name"] == "115_transformer", "count"].sum()

        # 3) count "115_incoming" in line_types by counting the ramaining "115_breaker" in drawing_components_list, and minus number of "115_breaker" in drawing_components_list with the number of "115_incoming" in line_types
        line_types_df.loc[line_types_df["name"] == "115_incoming", "count"] += drawing_components_df.loc[(
            drawing_components_df["name"] == "115_breaker"), "count"].sum()
        drawing_components_df.loc[drawing_components_df["name"] == "115_breaker",
                                  "count"] -= line_types_df.loc[line_types_df["name"] == "115_incoming", "count"].sum()

        # 4) count "22_tie" in line_types (default is 1) and minus 1 from the count of "22_breaker" in drawing_components_list
        line_types_df.loc[line_types_df["name"] == "22_tie", "count"] += 1
        drawing_components_df.loc[drawing_components_df["name"]
                                  == "22_breaker", "count"] -= 1

        # 5) count "22_capacitor" in line_types by counting "22_cap_bank" in drawing_components_list, and minus number of "22_breaker" in drawing_components_list with the number of "22_capacitor" in line_types
        line_types_df.loc[line_types_df["name"] == "22_capacitor", "count"] += drawing_components_df.loc[(
            drawing_components_df["name"] == "22_cap_bank"), "count"].sum()
        drawing_components_df.loc[drawing_components_df["name"] == "22_breaker",
                                  "count"] -= line_types_df.loc[line_types_df["name"] == "22_capacitor", "count"].sum()

        # 6) count "22_outgoing" in line_types by counting the "22_ds_la_out" or "22_ds_out" in drawing_components_list, and minus number of "22_breaker" in drawing_components_list with the number of "22_outgoing" in line_types
        line_types_df.loc[line_types_df["name"] == "22_outgoing", "count"] += drawing_components_df.loc[(
            drawing_components_df["name"] == "22_ds_la_out") | (drawing_components_df["name"] == "22_ds_out"), "count"].sum()
        drawing_components_df.loc[drawing_components_df["name"] == "22_breaker",
                                  "count"] -= line_types_df.loc[line_types_df["name"] == "22_outgoing", "count"].sum()

        # 7) count "22_incoming" in line_types by counting the ramaining "22_breaker" in drawing_components_list, and minus number of "22_breaker" in drawing_components_list with the number of "22_incoming" in line_types
        line_types_df.loc[line_types_df["name"] == "22_incoming", "count"] += drawing_components_df.loc[(
            drawing_components_df["name"] == "22_breaker"), "count"].sum()
        drawing_components_df.loc[drawing_components_df["name"] == "22_breaker",
                                  "count"] -= line_types_df.loc[line_types_df["name"] == "22_incoming", "count"].sum()

        # 8) count "22_service" in line_types by counting the "22_ds" in drawing_components_list
        line_types_df.loc[line_types_df["name"] == "22_service", "count"] += drawing_components_df.loc[(
            drawing_components_df["name"] == "22_ds"), "count"].sum()

        # 9) reset drawing_components_df to the original dataframe
        drawing_components_df = drawing_components_df_backup.copy()

        # # 10) remove the components with count = 0
        # drawing_components_df = drawing_components_df[drawing_components_df["count"]
        #                                               != 0].reset_index(drop=True)

        # # 99) conclue to the number of total 115_incoming, 115_transformer, 115_tie
        print(drawing_components_df)
        print(line_types_df)

    except Exception as e:
        print(e)

        return None, None

    finally:
        await prisma.disconnect()

        return drawing_components_df, line_types_df


async def valiateComponent(drawingType, drawing_components_df, line_types_df):
    try:
        print("Start validating component")

        drawingTruth = drawingTree[drawingType]

        # for each line type in 115 of line_types_df
        for index, row in line_types_df.iterrows():
            # get the line type name
            line_type_name = row["name"]
            # get the line type count
            line_type_count = row["count"]
            # get the line type mandatory components
            mandatories = drawingTruth[line_type_name]["mandatory"]

            # for each mandatory component in the line type
            for mandatory in mandatories:
                # check if the mandatory component has no children
                if isinstance(mandatories[mandatory], int):
                    # get the mandatory component count
                    mandatory_count = mandatories[mandatory]
                    # deduct the mandatory component count from the drawing_components_df
                    drawing_components_df.loc[drawing_components_df["name"]
                                              == mandatory, "count"] -= mandatory_count * line_type_count

                # means the mandatory component has children
                else:
                    # get the mandatory component children
                    mandatory_component_children = mandatories[
                        mandatory]

                    # get its _total truth
                    mandatory_component_children_total = mandatory_component_children["_total"] * \
                        line_type_count

                    while (mandatory_component_children_total > 0):
                        found = False

                        for child in mandatory_component_children:
                            if (child == "_total"):
                                continue

                            if(found):
                                break

                            # if there is a child in the drawing_components_df (not None)
                            if (drawing_components_df.loc[drawing_components_df["name"] == child, "count"].any()):
                                # deduct the child count from the drawing_components_df
                                drawing_components_df.loc[drawing_components_df["name"]
                                                            == child, "count"] -= 1 * line_type_count
                                # deduct the child count from the mandatory_component_children_total
                                mandatory_component_children_total -= 1 * \
                                    line_type_count
                                # set found to True
                                found = True

                        # # if there is no child in the drawing_components_df (None)
                        # if (not found):
                        #     raise Exception(
                        #         "There is no child in the drawing_components_df")


    except Exception as e:
        print(e)
        return None, None
    finally:
        print("Remaining drawing_components_df:", drawing_components_df)
        print("Finish validating component")


async def main() -> None:
    drawingType = "Main&Transfer"
    drawing_components_df, line_types_df = await countBay("e4a96435-bwa-BangWua1-sm-mt")
    await valiateComponent(drawingType, drawing_components_df, line_types_df)


if __name__ == '__main__':
    asyncio.run(main())
