import cv2
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import logging
from app.domain.service.tf_service import process_image, predict_mnist_digit

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 애플리케이션 시작/종료 시 모델 로딩/정리를 위한 lifespan 관리자
# @asynccontextmanager
# async def lifespan(app: APIRouter): # FastAPI 앱 대신 APIRouter에 적용 가능
# logger.info("TF-Service API 시작 - MNIST 모델 로딩 시도...")
# if load_mnist_model_globally():
# logger.info("MNIST 모델이 성공적으로 로드되었습니다 (애플리케이션 시작 시점).")
# else:
# logger.error("애플리케이션 시작 시 MNIST 모델 로딩 실패!")
# yield
# logger.info("TF-Service API 종료.")

router = APIRouter() # 기본 APIRouter 사용
# lifespan 관리자를 사용하기 위해 APIRouter 생성 시 적용
# router = APIRouter(lifespan=lifespan)

# 업로드된 파일을 저장할 디렉토리
UPLOAD_DIR = "./uploads"

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 업로드 디렉토리가 없으면 생성
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
            logger.info(f"'{UPLOAD_DIR}' 디렉토리를 생성했습니다.")

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"파일 '{file.filename}'이 '{file_path}' 경로에 성공적으로 업로드 및 저장되었습니다.")
        return JSONResponse(
            status_code=200,
            content={"message": "파일 업로드 성공", "filename": file.filename, "filepath": file_path}
        )
    except Exception as e:
        logger.error(f"파일 업로드 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"파일 업로드 실패: {str(e)}")
    
# 모자이크 처리 및 이미지 인식 엔드포인트
@router.post("/process")
async def process_uploaded_image(
    request_data: dict,  # JSON 본문으로 받도록 변경
    mosaic_size: int = Query(15, description="모자이크 블록 크기 (기본값: 15)")
):
    try:
        filename = request_data.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="파일명이 제공되지 않았습니다")
            
        # 업로드된 파일 경로 구성
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            logger.error(f"파일을 찾을 수 없음: {file_path}")
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없음: {filename}")
        
        # 이미지 처리 요청
        result = process_image(file_path, mosaic_size)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
            
        return JSONResponse(status_code=200, content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"이미지 처리 요청 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 처리 실패: {str(e)}")

# MNIST 숫자 예측 엔드포인트
@router.post("/predict", summary="MNIST 숫자 예측")
async def mnist_predict_endpoint(
    request_data: dict # JSON 본문으로 filename을 받음
):
    try:
        filename = request_data.get("filename")
        if not filename:
            logger.error("Predict endpoint: 'filename' not provided in request body.")
            raise HTTPException(status_code=400, detail="요청 본문에 'filename'이 제공되지 않았습니다.")

        # 업로드된 파일 경로 구성
        image_path = os.path.join(UPLOAD_DIR, filename)

        if not os.path.exists(image_path):
            logger.error(f"Predict endpoint: Image file '{image_path}' not found.")
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없음: {filename}")
        
        # MNIST 숫자 예측 서비스 호출
        logger.info(f"Predict endpoint: Requesting MNIST prediction for '{filename}'")
        prediction_result = predict_mnist_digit(image_path)
        
        if not prediction_result["success"]:
            # 서비스 함수에서 이미 로깅했을 것이므로, 여기서는 상태 코드와 메시지만 전달
            raise HTTPException(status_code=500, detail=prediction_result["message"])
            
        return JSONResponse(status_code=200, content=prediction_result)
    
    except HTTPException: # 이미 발생한 HTTPException은 그대로 전달
        raise
    except Exception as e:
        logger.error(f"Predict endpoint: Unexpected error during MNIST prediction for '{filename if 'filename' in locals() else '[unknown file]'}': {e}")
        raise HTTPException(status_code=500, detail=f"MNIST 예측 중 서버 오류 발생: {str(e)}")

# 모자이크된 이미지 다운로드 엔드포인트
@router.get("/download/{filename}")
async def download_processed_image(filename: str):
    try:
        # 모자이크 파일 경로 구성
        file_path = os.path.join("./mosaic", filename)
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            logger.error(f"파일을 찾을 수 없음: {file_path}")
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없음: {filename}")
        
        logger.info(f"모자이크 이미지 다운로드: {file_path}")
        return FileResponse(file_path, filename=filename)
    except Exception as e:
        logger.error(f"모자이크된 이미지 다운로드 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"모자이크된 이미지 다운로드 실패: {str(e)}")
    