import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# URI подключения к базе из переменных окружения
DB_URI = os.getenv("DB_URI", "postgresql+psycopg2://postgres:postgres@db:5432/musicdb")

# создаём движок SQLAlchemy
engine = create_engine(
    DB_URI,
    pool_pre_ping=True,
    future=True,
)

# фабрика сессий
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)

# базовый класс для всех моделей
class Base(DeclarativeBase):
    pass
