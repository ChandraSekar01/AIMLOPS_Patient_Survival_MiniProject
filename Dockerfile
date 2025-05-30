FROM python:3.11.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8082
CMD ["python", "app.py"]