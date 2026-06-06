FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Зөвхөн кодыг хуулна — дата/загвар volume-оор mount хийнэ
COPY api/ ./api/
COPY core/ ./core/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
