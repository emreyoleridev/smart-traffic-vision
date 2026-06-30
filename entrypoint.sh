#!/bin/bash

PORT=${PORT:-8501}

# Start FastAPI backend in the background on a fixed internal port
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend on the external PORT provided by Render
streamlit run streamlit_app/app.py --server.port $PORT --server.address 0.0.0.0
