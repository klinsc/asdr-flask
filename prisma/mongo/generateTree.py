# this is the tree map for "Main&Transfer" drawing
import pymongo
import asyncio
from drawingType.drawing import drawing

async def main() -> None:

    client = pymongo.MongoClient("mongodb://localhost:27017/")

    document = client["drawingType"]

    collection = document["Main&Transfer"]

    collection.insert_one(drawing["Main&Transfer"])

if __name__ == '__main__':
    asyncio.run(main())
