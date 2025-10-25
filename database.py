from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

Base = declarative_base()

class WorkoutSession(Base):
    __tablename__ = "workout_sessions"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sets = relationship("WorkoutSet", back_populates="session")

class WorkoutSet(Base):
    __tablename__ = "workout_sets"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("workout_sessions.id"))
    exercise = Column(String)
    good_reps = Column(Integer)
    bad_reps = Column(Integer)
    session = relationship("WorkoutSession", back_populates="sets")

# Create and connect
engine = create_engine("sqlite:///workouts.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
