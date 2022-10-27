#!/bin/sh
export FLASK_APP=./python_script.py
source .venv/bin/activate
flask run -h 0.0.0.0