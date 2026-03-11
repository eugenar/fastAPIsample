from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

db_file_name = "database.db"
db_url = f"sqlite+aiosqlite:///{db_file_name}"
connect_args = {"check_same_thread": False}


engine = create_async_engine(db_url, connect_args=connect_args)
db_file_name = "database.db"


async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)