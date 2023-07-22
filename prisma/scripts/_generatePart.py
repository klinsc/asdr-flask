import asyncio
from prisma import Prisma


async def main() -> None:
    prisma = Prisma()
    await prisma.connect()

    # write your queries here
    await prisma.part.create_many(
        skip_duplicates=True,
        data=[
            {
                "name": "kv115",
            },
            {
                "name": "kv22",
            },
            {
                "name": "universal",
            },
        ]
    )

    await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
