#!/usr/bin/env python

from dotenv import load_dotenv
from flask import Flask
import os

app = Flask(__name__, static_folder='static')
load_dotenv('../.env')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", None)

from app import routes
