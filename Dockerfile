FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

# Hugging Face Spaces 需要暴露端口 7860
EXPOSE 7860

CMD ["python", "bot.py"]
