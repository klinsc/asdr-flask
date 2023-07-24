import asyncio
from prisma import Prisma


async def main() -> None:
    prisma = Prisma()
    await prisma.connect()

    drawingType = "Main&Transfer"
    drawingTypeId = await prisma.drawingtype.find_unique(
        where={
            "name": drawingType
        }
    )
    print(drawingTypeId)

    # write your queries here
    await prisma.drawing.create(
        data={
            "name": "e4a96435-bwa-BangWua1-sm-mt",
            "drawingTypeId": drawingTypeId.id,
        }
    )

    await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
