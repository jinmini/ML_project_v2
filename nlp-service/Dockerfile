FROM python:3.12.7-slim

# 시스템 패키지 업데이트 및 OpenJDK 11 (또는 다른 호환 버전) 설치
# noninteractive 설정을 추가하여 빌드 중 사용자 입력 방지
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
# pip install 전 setuptools 업데이트 (간혹 필요할 수 있음)
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9003", "--reload"] 