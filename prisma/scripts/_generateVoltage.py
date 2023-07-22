import asyncio
from prisma import Prisma

async def main() -> None:
    prisma = Prisma()
    await prisma.connect()

    # write your queries here
    await prisma.voltage.create_many(
        data=[
            {
                "name": "115",
            },
            {
                "name": "22",
            },
            {
                "name": "33",
            },
        ]
    )

    await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
