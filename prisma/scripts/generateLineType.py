import asyncio
from prisma import Prisma


async def main() -> None:
    prisma = Prisma()
    await prisma.connect()

    drawingType = await prisma.drawingtype.find_unique(
        where={
            "name": "Main&Transfer"
        }
    )
    if (drawingType == None):
        raise Exception("Drawing not found")

    # write your queries here
    await prisma.linetype.create_many(
        skip_duplicates=True,
        data=[
            {
                "name": "115_incoming",
                "drawingTypeId": drawingType.id
            },
            {
                "name": "115_transformer",
                "drawingTypeId": drawingType.id
            },
            {
                "name": "115_tie",
                "drawingTypeId": drawingType.id
            },
            {
                "name": "22_capacitor",
                "drawingTypeId": drawingType.id
            },
            {
                "name": "22_feeder",
                "drawingTypeId": drawingType.id
            },
            {
                "name": "22_incoming",
                "drawingTypeId": drawingType.id
            },
            {
                "name": "22_outgoing",
                "drawingTypeId": drawingType.id
            },
            {
                "name": "22_tie",
                "drawingTypeId": drawingType.id
            },
        ]
    )

    await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
