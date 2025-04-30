# python -m app.main

from app.domain.controller.titanic_controller import Controller
import time
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from dotenv import load_dotenv
from app.api.titanic_router import router as titanic_router

def print_header():
    """타이틀 헤더를 출력하는 함수"""
    os.system('cls' if os.name == 'nt' else 'clear')  # 화면 지우기
    
    print("\n" + "="*70)
    print("🚢 타이타닉 생존자 예측 모델 🚢".center(70))
    print("="*70)
    print("이 프로그램은 타이타닉 승객 데이터를 기반으로 생존 여부를 예측합니다.")
    print("다양한 머신러닝 알고리즘을 비교하여 최적의 모델을 찾습니다.")
    print("="*70 + "\n")

def main():
    """타이타닉 생존 예측 모델 실행 함수"""
    print_header()
    
    # 컨트롤러 초기화
    print("🔄 시스템 초기화 중...")
    controller = Controller()
    
    # 파일 경로 설정
    train_data = "train.csv"
    test_data = "test.csv"
    
    # 데이터 전처리
    print("\n🔄 데이터 전처리 시작...")
    time.sleep(1)  # 시각적 효과를 위한 지연
    controller.preprocess(train_data, test_data)
    
    # 모델 학습
    print("\n🔄 모델 학습 시작...")
    time.sleep(1)  # 시각적 효과를 위한 지연
    controller.learning()
    
    # 모델 평가
    print("\n🔄 모델 평가 시작...")
    time.sleep(1)  # 시각적 효과를 위한 지연
    controller.evaluation()
    
    # 결과 제출
    print("\n🔄 최종 결과 생성...")
    time.sleep(1)  # 시각적 효과를 위한 지연
    submission = controller.submit()
    
    if submission is not None:
        print("\n📋 제출 데이터 미리보기:")
        print(submission.head())
        
        # 제출 파일 경로 확인
        submission_path = os.path.abspath("./updated_data/submission.csv")
        print(f"\n📂 제출 파일 경로: {submission_path}")
        print(f"✓ 이 파일을 Kaggle 타이타닉 대회에 제출하세요!")
    
    # 프로그램 종료
    print("\n" + "="*70)
    print("✅ 타이타닉 생존자 예측 모델 분석이 완료되었습니다! ✅".center(70))
    print("="*70)
    return submission

# ✅로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("titanic_api")

# ✅.env 파일 로드
load_dotenv()

# ✅ 애플리케이션 시작 시 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Titanic API 서비스 시작")
    yield
    logger.info("🛑 Titanic API 서비스 종료")

# ✅ FastAPI 앱 생성 
app = FastAPI(
    title="Titanic API",
    description="Titanic API for jinmini.com",
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
app.include_router(titanic_router, prefix="/titanic")

# ✅ 서버 실행
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9001))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)



