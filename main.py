from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import time
from typing import Annotated, Union

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from api_client import api_client
from database import create_db_and_tables, engine
from models import Note, Patient


class Settings(BaseSettings):
    OPENAI_API_KEY: str


async def get_session():
    async with AsyncSession(engine) as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],  # Allows all standard methods
    allow_headers=["*"],  # Allows all standard headers
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Logs the request path and processing time.
    """
    start_time = time.perf_counter()

    # Forward request to route handler
    response = await call_next(request)

    process_time = time.perf_counter() - start_time
    print(
        f"Request: {request.url.path} completed in {process_time:.4f} seconds with status {response.status_code}"
    )

    return response


@app.post("/patients/")
async def create_patient(patient: Patient, session: SessionDep) -> Patient:
    patient.id = None  # Ensure a new patient is created
    patient.date_of_birth = datetime.strptime(patient.date_of_birth, "%Y-%m-%d").date()
    db_patient = Patient.model_validate(patient)
    session.add(db_patient)
    await session.commit()
    await session.refresh(db_patient)
    return db_patient


@app.get("/patients/")
async def read_patients(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Patient]:
    patients = session.exec(select(Patient).offset(offset).limit(limit)).all()
    return patients


@app.get("/patients/{id}")
async def read_patient(id: int, session: SessionDep) -> Patient:
    patient = session.get(Patient, id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.patch("/patients/{id}")
async def update_patient(id: int, patient: Patient, session: SessionDep):
    patient_db = session.get(Patient, id)
    if not patient_db:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient.date_of_birth = datetime.strptime(patient.date_of_birth, "%Y-%m-%d").date()
    patient.id = id  # Ensure we are updating the correct patient
    patient_db.sqlmodel_update(patient)
    session.add(patient_db)
    await session.commit()
    await session.refresh(patient_db)
    return patient_db


@app.delete("/patients/{id}")
async def delete_patient(id: int, session: SessionDep):
    patient = session.get(Patient, id)
    if patient:
        await session.delete(patient)
        await session.commit()
    return {"ok": True}


@app.post("/notes/{patient_id}")
async def create_note(patient_id: int, note: Note, session: SessionDep) -> Note:
    note.id = None  # Ensure a new note is created
    note.patient_id = patient_id
    note.update_date = datetime.now(timezone.utc)
    db_note = Note.model_validate(note)
    session.add(db_note)
    await session.commit()
    await session.refresh(db_note)
    return db_note


@app.get("/notes/{patient_id}")
async def read_notes(
    patient_id: int,
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Note]:
    notes = session.exec(
        select(Note).where(Note.patient_id == patient_id).offset(offset).limit(limit)
    ).all()
    return notes


@app.patch("/notes/{id}")
async def update_note(id: int, note: Note, session: SessionDep):
    note_db = await session.get(Note, id)
    if not note_db:
        raise HTTPException(status_code=404, detail="Note not found")

    note.update_date = datetime.now(timezone.utc)
    note.id = id  # Ensure we are updating the correct note
    note.patient_id = note_db.patient_id  # Prevent changing patient_id
    note_db.sqlmodel_update(note)
    session.add(note_db)
    await session.commit()
    await session.refresh(note_db)
    return note_db


@app.delete("/notes/{id}")
async def delete_note(id: int, session: SessionDep):
    note = await session.get(Note, id)
    if note:
        await session.delete(note)
        await session.commit()
    return {"ok": True}


@app.get(
    "/notes/summary/{patient_id}",
    summary="Summarize notes for a given patient using a client API (like OpenAI)",
)
async def read_notes_summary(patient_id: int, session: SessionDep):
    patient = await session.get(Patient, patient_id)
    notes = await session.exec(select(Note).where(Note.patient_id == patient_id)).all()

    # Combine all note contents into a single string
    combined_text = " ".join(note.content for note in notes)

    try:
        summary = await api_client.get_summary(combined_text)

        return json.dumps(
            {
                "patient_id": patient_id,
                "name": patient.name,
                "date_of_birth": patient.date_of_birth,
                "summary": summary,
            },
            default=str,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while summarizing notes: {str(e)}",
        )


@app.get("/health")
async def health():
    return {"status": "ok"}
