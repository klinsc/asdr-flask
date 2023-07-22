import asyncio
from prisma import Prisma

async def main() -> None:
    prisma = Prisma()
    await prisma.connect()

    # write your queries here
    await prisma.part.create_many(
        data=[
            {
                "name": "kv115",
            },
            {
                "name": "kv22",
            },
            {
                "name": "Universal",
            },
        ]
    )

    await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
