import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api.nlp_router import router as nlp_router
from app.utils.logger import logger # 설정된 로거 임포트

# .env 파일 로드
load_dotenv()

# 애플리케이션 시작/종료 시 실행될 로직 (lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 NLP Service 시작")
    yield
    logger.info("🛑 NLP Service 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="NLP Service",
    description="텍스트 분석 및 처리를 위한 NLP 서비스 API (Samsung Report Wordcloud)",
    version="0.1.1", # 버전 업데이트
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
    return {"message": "Welcome to NLP Service"}

# NLP 관련 라우터 포함
app.include_router(nlp_router, prefix="/nlp")
logger.info("NLP 라우터 등록 완료")

# 서버 직접 실행 시 (uvicorn 사용)
if __name__ == "__main__":
    import uvicorn
    # .env 파일에서 포트 가져오기 (없으면 9003 기본값)
    port = int(os.getenv("PORT", 9003))
    logger.info(f"서버 시작 - Host: 0.0.0.0, Port: {port}")
    # reload=True는 개발 중에만 사용하는 것이 좋음
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)





