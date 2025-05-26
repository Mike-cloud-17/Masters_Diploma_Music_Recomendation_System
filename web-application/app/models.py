from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .db import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    msno       = Column(String, unique=True, index=True)  # hash of profile
    profile    = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    recos      = relationship("Recommendation", back_populates="user")
    fb         = relationship("Feedback", back_populates="user")

class Recommendation(Base):
    __tablename__ = "recommendations"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    user_id   = Column(Integer, ForeignKey("users.id"))
    track_id  = Column(Integer)
    rank      = Column(Integer)
    played    = Column(Boolean, default=False)

    user      = relationship("User", back_populates="recos")

class Feedback(Base):
    __tablename__ = "feedback"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    user_id   = Column(Integer, ForeignKey("users.id"))
    track_id  = Column(Integer)
    liked     = Column(Boolean)
    ts        = Column(DateTime, default=datetime.utcnow)

    user      = relationship("User", back_populates="fb")
