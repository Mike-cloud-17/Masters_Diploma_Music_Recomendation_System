from flask import Flask
from .db import Base, engine
from .routes import register_routes

def create_app() -> Flask:
    Base.metadata.create_all(engine)
    app = Flask(__name__)
    register_routes(app)
    return app
