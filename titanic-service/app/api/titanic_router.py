from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from app.domain.model.titanic_schema import (
    TitanicRequest,
    TitanicDataResponse
)
from app.domain.controller.titanic_controller import Controller

# 로거 설정
logger = logging.getLogger("titanic_router")
logger.setLevel(logging.INFO)
router = APIRouter()

@router.post("/submit", response_model=TitanicDataResponse)
async def titanic(request: TitanicRequest):
    logger.info(f"Titanic request: {request}")
    
    try:
        # 컨트롤러 초기화
        controller = Controller()
        
        # 데이터 전처리
        logger.info("타이타닉 데이터 전처리 시작...")
        controller.preprocess("train.csv", "test.csv")
        
        # 모델 학습
        logger.info("타이타닉 모델 학습 시작...")
        controller.learning()
        
        # 모델 평가
        logger.info("타이타닉 모델 평가 시작...")
        best_accuracy, best_model = controller.evaluation()
        
        # 결과 제출
        logger.info("타이타닉 예측 결과 생성 및 CSV 파일 저장...")
        submission = controller.submit()
        
        # 예측 결과 처리
        if submission is not None:
            predictions_list = []
            # 일부 예측 결과만 API 응답에 포함 (최대 10개)
            for idx, row in submission.head(10).iterrows():
                predictions_list.append({
                    "passenger_id": int(row["PassengerId"]),
                    "survived": int(row["Survived"])
                })
            
            survived_count = int(submission["Survived"].sum())
            total_passengers = len(submission)
            
            return TitanicDataResponse(
                message="모델평가가 진행되었고 파일이 저장되었습니다!👍☺️",
                success=True,
                predictions=predictions_list,
                total_passengers=total_passengers,
                survived_count=survived_count,
                death_count=total_passengers - survived_count
            )
        else:
            return TitanicDataResponse(
                message="타이타닉 예측 실패 - 모델 평가 완료 안됨",
                success=False
            )
    except Exception as e:
        logger.error(f"Error processing titanic request: {e}")
        return TitanicDataResponse(
            message=f"오류 발생: {str(e)}",
            success=False
        )
