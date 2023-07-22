import asyncio
from prisma import Prisma

async def main() -> None:
    prisma = Prisma()
    await prisma.connect()

    # write your queries here
    await prisma.drawingtype.create_many(
        data=[
            {
                "name": "Main&Transfer",
            },
            {
                "name": "H-config",
            },
            {
                "name": "DoubleBusSingleBreaker",
            },
            {
                "name": "Breaker&aHalf",
            },
        ]
    )

    await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
