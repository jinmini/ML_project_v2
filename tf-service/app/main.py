import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api.tf_router import router as tf_router
from app.utils.logger import logger # 설정된 로거 임포트
# MNIST 모델 로더 함수 임포트
from app.domain.service.tf_service import load_mnist_model_globally

# .env 파일 로드
load_dotenv()

# 애플리케이션 시작/종료 시 실행될 로직 (lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 TF Service 시작")
    logger.info("MNIST 모델 로딩 시도...")
    if load_mnist_model_globally():
        logger.info("MNIST 모델이 성공적으로 전역 로드되었습니다.")
    else:
        logger.error("MNIST 모델 전역 로딩 실패! 예측 기능이 작동하지 않을 수 있습니다.")
    yield
    logger.info("🛑 TF Service 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="TF Service",
    description="이미지 처리 및 MNIST 숫자 예측을 위한 TF 서비스 API", # 설명 업데이트
    version="0.1.2", # 버전 업데이트
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 오리진 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"], # 모든 HTTP 메소드 허용
    allow_headers=["*"], # 모든 헤더 허용
)

# 기본 경로 (루트)
@app.get("/")
async def read_root():
    logger.info("메인 루트 경로 호출됨")
    return {"message": "Welcome to TF Service for Image Processing and MNIST Prediction"} # 메시지 업데이트

# TF 관련 라우터 포함
app.include_router(tf_router, prefix="/tf")
logger.info("TF 라우터 등록 완료 (/tf 접두사 사용)")