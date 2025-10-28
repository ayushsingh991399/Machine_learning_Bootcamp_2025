FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY predict2.py ./

RUN pip install --no-cache-dir fastapi uvicorn pydantic scikit-learn

EXPOSE 3000

CMD ["uvicorn", "predict2:app", "--host", "0.0.0.0", "--port", "3000"]
