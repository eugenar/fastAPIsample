from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

import config

db_url = config["database"]["parameters"]["url"]

if config["database"]["type"] == "sqlite":
    connect_args = {
        "check_same_thread": False
    }  # SQLite-specific argument to allow connections from multiple threads
else:
    connect_args = None

engine = create_async_engine(db_url, connect_args=connect_args)


async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
