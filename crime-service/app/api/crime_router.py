from fastapi import APIRouter, HTTPException
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

# --- 로깅 핸들러 설정 (애플리케이션 레벨에서 설정하는 것이 더 일반적) ---
# 만약 애플리케이션 레벨 설정이 없다면 여기에 추가:
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# ---------------------------------------------------------------

router = APIRouter()

@router.get("/details", summary="범죄 상세 정보 및 상관계수 분석 결과 조회")
async def get_details():
    logger.info("상관계수 분석 요청 받음")
    try:
        controller = CrimeController()
        # get_correlation_results 내부에서 발생하는 예외 처리 필요
        correlation_results = controller.get_correlation_results()

        # 결과가 없을 경우 처리 (선택적)
        if not correlation_results or not correlation_results.get('districts'):
             logger.warning("상관계수 분석 결과 데이터가 비어있습니다.")
             # 빈 데이터를 반환하거나 404 에러 반환 선택
             # return CrimeDataResponse(message="분석 결과 데이터가 없습니다.", success=False)
             raise HTTPException(status_code=404, detail="상관계수 분석 결과 데이터가 없습니다.")

        total_districts = len(correlation_results.get('districts', []))
        logger.info(f"상관계수 분석 완료. 총 {total_districts}개 지역구 데이터 반환.")
        return CrimeDataResponse(
            message="상관계수 분석이 성공적으로 완료되었습니다.",
            success=True,
            crime_data=correlation_results,
            total_districts=total_districts
        )
    except HTTPException as e:
         # Controller 또는 Service에서 발생한 HTTPException 처리
         logger.error(f"상관계수 분석 중 HTTP 오류 발생: {e.status_code} - {e.detail}")
         # HTTPException은 그대로 다시 발생시켜 FastAPI가 처리하도록 함
         raise e
    except Exception as e:
        # 예상치 못한 서버 오류 처리
        logger.error(f"상관계수 분석 중 예상치 못한 오류 발생: {str(e)}", exc_info=True) # exc_info=True로 트레이스백 로깅
        # 클라이언트에게는 일반적인 500 오류 메시지 반환
        raise HTTPException(status_code=500, detail=f"상관계수 분석 처리 중 서버 오류 발생")

@router.get("/map", summary="범죄 지도 생성 및 제공")
async def get_crime_map():
    logger.info("범죄 지도 생성 요청 받음")
    try:
        controller = CrimeController()
        result = controller.draw_crime_map()
        file_path = result.get('file_path')
        if not file_path or not os.path.exists(file_path):
            logger.error(f"지도 생성은 성공했으나 결과 파일을 찾을 수 없음: {file_path}")
            raise HTTPException(status_code=404, detail="생성된 지도 파일을 찾을 수 없습니다.")

        logger.info(f"범죄 지도 파일 반환: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            # 성공적으로 HTML 파일 읽기 완료
            return HTMLResponse(content=html_content)
        except Exception as e:
            # 파일 읽기 오류 처리
            logger.error(f"지도 파일({file_path}) 읽기 오류: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="지도 파일을 읽는 중 오류가 발생했습니다.")

    except HTTPException as e:
        # Controller -> Service -> CrimeMapCreator 에서 발생시킨 HTTPException 처리
        logger.error(f"범죄 지도 생성 중 HTTP 오류 발생: {e.status_code} - {e.detail}")
        raise e # FastAPI가 적절한 JSON 응답 생성

    except Exception as e:
        # 예상치 못한 기타 오류 처리
        logger.error(f"범죄 지도 생성 중 예상치 못한 오류 발생: {str(e)}", exc_info=True)
        # 클라이언트에게는 일반적인 500 오류 메시지 반환
        raise HTTPException(status_code=500, detail="범죄 지도 생성 중 서버 오류가 발생했습니다.")

@router.post("/submit", response_model=CrimeDataResponse)
async def crime(request: CrimeRequest):
    # 이 엔드포인트는 현재 실제 로직이 없으므로 그대로 둠
    logger.info(f"Crime request: {request}")
    return CrimeDataResponse(message="Crime request received")

