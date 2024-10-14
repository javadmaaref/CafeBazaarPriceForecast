# models/database_models.py


from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class App(Base):
    __tablename__ = 'apps'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    category = Column(String)
    rating = Column(Float)
    reviews = Column(Integer)
    size = Column(Float)
    installs = Column(Integer)
    price = Column(Float)

    def __repr__(self):
        return f"<App(name='{self.name}', category='{self.category}', price={self.price})>"