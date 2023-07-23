import asyncio
from prisma import Prisma
import pandas as pd


async def main() -> None:
    try:
        prisma = Prisma()
        await prisma.connect()

        drawing_name = "e4a96435-bwa-BangWua1-sm-mt"

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

        # >>>>>>>>>
        # count lines in the drawing
        # 1) count "115_tie" in line_types (default is 1) and minus 1 from the count of "115_breaker" in drawing_components_list
        line_types_df.loc[line_types_df["name"] == "115_tie", "count"] += 1
        drawing_components_df.loc[drawing_components_df["name"]
                                  == "115_breaker", "count"] -= 1

        # 2) count "115_transformer" in line_types by counting "11522_tx_dyn1" or "11522_tx_ynyn0d1" in drawing_components_list, and minus number of "115_breaker" in drawing_components_list with the number of "115_transformer" in line_types
        line_types_df.loc[line_types_df["name"] == "115_transformer", "count"] += drawing_components_df.loc[(
            drawing_components_df["name"] == "11522_tx_dyn1") | (drawing_components_df["name"] == "11522_tx_ynyn0d1"), "count"].sum()
        drawing_components_df.loc[drawing_components_df["name"] == "115_breaker",
                                  "count"] -= line_types_df.loc[line_types_df["name"] == "115_transformer", "count"].sum()

        # 3) count "115_incoming" in line_types by counting the ramaining "115_breaker" in drawing_components_list
        line_types_df.loc[line_types_df["name"] == "115_incoming", "count"] += drawing_components_df.loc[(
            drawing_components_df["name"] == "115_breaker"), "count"].sum()

        # 4) conclue to the number of total 115_incoming, 115_transformer, 115_tie
        print(drawing_components_df)
        print(line_types_df)


    except Exception as e:
        print(e)

    finally:
        await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())

 # drawing_components_df.loc[drawing_components_df["name"] == "115_breaker", "count"] -= 1
    # # 2) minus 2 from the count of "115_ds" in drawing_components_list
    # drawing_components_df.loc[drawing_components_df["name"] == "115_ds", "count"] -= 2
