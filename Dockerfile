FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
COPY src/ ./src/
COPY saved_models/ ./saved_models/  
RUN pip install -r requirements.txt
CMD ["python", "./src/inference.py"]