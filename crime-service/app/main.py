# python -m app.main

import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from dotenv import load_dotenv
from app.api.crime_router import router as crime_router
from app.domain.controller.crime_controller import CrimeController

# ✅로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("crime_api")

# ✅.env 파일 로드
load_dotenv()

# ✅ 애플리케이션 시작 시 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Crime API 서비스 시작")
    
    # CrimeController 테스트 실행
    await test_crime_controller()
    
    yield
    logger.info("🛑 Crime API 서비스 종료")

# CrimeController 테스트 함수
async def test_crime_controller():
    logger.info("\n===== CrimeController 테스트 시작 =====\n")
    
    try:
        # CrimeController 인스턴스 생성
        controller = CrimeController()
        
        # 상관계수 분석 수행
        logger.info("\n===== 상관계수 분석 시작 =====")
        controller.correlation()
        logger.info("\n===== 상관계수 분석 완료 =====")
        
        logger.info("\n===== CrimeController 테스트 완료 =====\n")
    except Exception as e:
        logger.error(f"CrimeController 테스트 중 오류 발생: {str(e)}")

# ✅ FastAPI 앱 생성 
app = FastAPI(
    title="Crime API",
    description="Crime API for jinmini.com",
    version="0.1.0",
    lifespan=lifespan
)

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 서브 라우터와 엔드포인트를 연결
app.include_router(crime_router, prefix="/crime")

# ✅ 서버 실행
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9002))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)




