from typing import Union

from sqlmodel import Field, SQLModel


from datetime import date, datetime, timezone


class Patient(SQLModel, table=True):
    id: Union[int, None] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    date_of_birth: date


class Note(SQLModel, table=True):
    id: Union[int, None] = Field(default=None, primary_key=True)
    content: str
    patient_id: int = Field(foreign_key="patient.id", index=True)
    update_date: datetime = Field(default_factory=datetime.now(timezone.utc))