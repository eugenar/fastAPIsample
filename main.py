from datetime import date, datetime, timezone
import json
from typing import Annotated, Union

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic_settings import BaseSettings
from openai import OpenAI
from sqlmodel import Field, Session, SQLModel, create_engine, select

class Settings(BaseSettings):
    OPENAI_API_KEY: str

class Patient(SQLModel, table=True):
    id: Union[int, None] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    date_of_birth: date

class Note(SQLModel, table=True):
    id: Union[int, None] = Field(default=None, primary_key=True)
    content: str
    patient_id: int = Field(foreign_key="patient.id", index=True)
    update_date: datetime = Field(default_factory=datetime.now(timezone.utc))


db_file_name = "database.db"
db_url = f"sqlite:///{db_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(db_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI()

# Initialize the OpenAI client (it automatically picks up the OPENAI_API_KEY environment variable)
client = OpenAI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.post("/patients/")
def create_patient(patient: Patient, session: SessionDep) -> Patient:
    patient.id = None  # Ensure a new patient is created
    patient.date_of_birth = datetime.strptime(patient.date_of_birth, "%Y-%m-%d").date()
    db_patient = Patient.model_validate(patient)
    session.add(db_patient)
    session.commit()
    session.refresh(db_patient)
    return db_patient

@app.get("/patients/")
def read_patients(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Patient]:
    patients = session.exec(select(Patient).offset(offset).limit(limit)).all()
    return patients

@app.get("/patients/{id}")
def read_patient(id: int, session: SessionDep) -> Patient:
    patient = session.get(Patient, id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@app.patch("/patients/{id}")
def update_patient(id: int, patient: Patient, session: SessionDep):
    patient_db = session.get(Patient, id)
    if not patient_db:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient.date_of_birth = datetime.strptime(patient.date_of_birth, "%Y-%m-%d").date()
    patient.id = id # Ensure we are updating the correct patient
    patient_db.sqlmodel_update(patient)
    session.add(patient_db)
    session.commit()
    session.refresh(patient_db)
    return patient_db

@app.delete("/patients/{id}")
def delete_patient(id: int, session: SessionDep):
    patient = session.get(Patient, id)
    if patient:
        session.delete(patient)
        session.commit()
    return {"ok": True}


@app.post("/notes/{patient_id}")
def create_note(patient_id: int, note: Note, session: SessionDep) -> Note:
    note.id = None  # Ensure a new note is created
    note.patient_id = patient_id 
    note.update_date = datetime.now(timezone.utc)
    db_note = Note.model_validate(note)
    session.add(db_note)
    session.commit()
    session.refresh(db_note)
    return db_note

@app.get("/notes/{patient_id}")
def read_notes(
    patient_id: int,
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Note]:
    notes = session.exec(select(Note).where(Note.patient_id == patient_id).offset(offset).limit(limit)).all()
    return notes

@app.patch("/notes/{id}")
def update_note(id: int, note: Note, session: SessionDep):
    note_db = session.get(Note, id)
    if not note_db:
        raise HTTPException(status_code=404, detail="Note not found")

    note.update_date = datetime.now(timezone.utc)
    note.id = id # Ensure we are updating the correct note
    note.patient_id = note_db.patient_id # Prevent changing patient_id
    note_db.sqlmodel_update(note)
    session.add(note_db)
    session.commit()
    session.refresh(note_db)
    return note_db

@app.delete("/notes/{id}")
def delete_note(id: int, session: SessionDep):
    note = session.get(Note, id)
    if note:
        session.delete(note)
        session.commit()
    return {"ok": True}

@app.get("/notes/summary/{patient_id}", summary="Summarize notes for a given patient using an OpenAI LLM")
def read_notes_summary(
    patient_id: int,
    session: SessionDep):
    patient = session.get(Patient, patient_id)
    notes = session.exec(select(Note).where(Note.patient_id == patient_id)).all()

    # Combine all note contents into a single string
    combined_text = " ".join(note.content for note in notes)

    try:
        # Call the OpenAI API for chat completions to summarize the notes
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and concise summarizer."},
                {"role": "user", "content": f"Summarize this text in one paragraph: {combined_text}"}
            ],
            max_completion_tokens=100
        )

        # Extract the summary content from the response
        summary = response.choices[0].message.content.strip()
        return json.dumps({"patient_id": patient_id, "name": patient.name, "date_of_birth": patient.date_of_birth, "summary": summary}, default=str)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while summarizing notes: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}