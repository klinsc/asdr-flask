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
        components = await prisma.component.find_many()
        if (len(components) == 0):
            raise Exception("Component not found")

        # create a dataframe to store the data
        df = pd.DataFrame(columns=["id", "index", "name", "count"])

        # loop through all components
        for component in components:
            df = df._append({"id": component.id, "index": component.index,
                            "name": component.name, "count": 0}, ignore_index=True)

        with open(f"./txt/{drawing_name}.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                # split line by space
                line = line.split()
                # get class index and parse to int
                classIndex = int(line[0])

                # count the number of components in each class
                for index, row in df.iterrows():
                    row_index = row["index"]
                    if row_index == classIndex:
                        # update the count in the dataframe
                        new_count = row["count"] + 1
                        df.loc[index, "count"] = new_count

            # save the count in the database on the drawingComponent table
            # using bulk create to save the count in the database
            # https://www.prisma.io/docs/concepts/components/prisma-client/crud#create-multiple-records
            await prisma.drawingcomponent.create_many(
                data=[
                    {
                        "drawingId": drawing.id,
                        "componentId": row["id"],
                        "count": row["count"]
                    }
                    for index, row in df.iterrows()
                ]
            )

    except Exception as e:
        print(e)

    finally:
        await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
