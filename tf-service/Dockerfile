FROM python:3.12.7-slim

WORKDIR /app

# OpenCV 실행에 필요한 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9004", "--reload"] 