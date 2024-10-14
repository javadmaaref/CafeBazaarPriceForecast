# models/setup_database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database_models import Base, App
from config import DATABASE_URI

def setup_database():
    engine = create_engine(DATABASE_URI)
    Base.metadata.create_all(engine)
    print("Database and tables created successfully.")
    return engine

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()

if __name__ == "__main__":
    setup_database()
