from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from app.domain.model.crime_schema import (
    CrimeRequest,
    CrimeDataResponse
)
from app.domain.controller.crime_controller import CrimeController
from fastapi.responses import HTMLResponse, JSONResponse
import os

# 로거 설정
logger = logging.getLogger("crime_router")
logger.setLevel(logging.INFO)
router = APIRouter()

@router.get("/details", summary="범죄 상세 정보 및 상관계수 분석 결과 조회")
async def get_details():
    logger.info("상관계수 분석 요청 받음")
    
    try:
        # CrimeController 인스턴스 생성
        controller = CrimeController()
        
        # 상관계수 분석 수행 및 결과 반환
        correlation_results = controller.get_correlation_results()
        
        # 지역구 개수 계산
        total_districts = len(correlation_results.get('districts', []))
        
        # 응답 구성
        return CrimeDataResponse(
            message="상관계수 분석이 성공적으로 완료되었습니다.",
            success=True,
            crime_data=correlation_results,
            total_districts=total_districts
        )
    except Exception as e:
        logger.error(f"상관계수 분석 중 오류 발생: {str(e)}")
        return CrimeDataResponse(
            message=f"상관계수 분석 중 오류 발생: {str(e)}",
            success=False
        )

@router.get("/map", summary="범죄 지도 생성 및 제공")
async def get_crime_map():
    logger.info("범죄 지도 생성 요청 받음")
    
    try:
        # CrimeController 인스턴스 생성
        controller = CrimeController()
        
        # 범죄 지도 생성
        result = controller.draw_crime_map()
        
        if result.get("status") == "success":
            file_path = result.get("file_path")
            if os.path.exists(file_path):
                logger.info(f"범죄 지도 파일 반환: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    return HTMLResponse(content=html_content)
                except Exception as e:
                    logger.error(f"지도 파일 읽기 오류: {str(e)}")
                    return JSONResponse(
                        content={"message": f"지도 파일 읽기 오류: {str(e)}", "success": False},
                        status_code=500
                    )
            else:
                logger.error(f"생성된 지도 파일을 찾을 수 없음: {file_path}")
                return JSONResponse(
                    content={"message": f"생성된 지도 파일을 찾을 수 없습니다: {file_path}", "success": False},
                    status_code=404
                )
        else:
            logger.error(f"범죄 지도 생성 실패: {result.get('message')}")
            return JSONResponse(
                content={"message": f"범죄 지도 생성 중 오류 발생: {result.get('message')}", "success": False},
                status_code=500
            )
    except Exception as e:
        logger.error(f"범죄 지도 생성 중 오류 발생: {str(e)}")
        return JSONResponse(
            content={"message": f"범죄 지도 생성 중 오류 발생: {str(e)}", "success": False},
            status_code=500
        )

@router.post("/submit", response_model=CrimeDataResponse)
async def crime(request: CrimeRequest):
    logger.info(f"Crime request: {request}")
    return CrimeDataResponse(message="Crime request received")

