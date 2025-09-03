.PHONY: setup build app clean

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

build:
\t. .venv/bin/activate && python -m src.cli.main build-all

app:
\t. .venv/bin/activate && streamlit run src/app/streamlit_app.py

clean:
\trm -rf data/processed reports/exports reports/logs || true