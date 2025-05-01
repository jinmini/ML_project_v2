import traceback
import os
import app.domain.service.internal.map_create as map_creator
from fastapi import HTTPException 
import logging 

logger = logging.getLogger(__name__) 

class CrimeVisualizer:

    def __init__(self):
        pass 
        
    def draw_crime_map(self) -> dict:
        """범죄 지도를 생성하고 결과를 반환합니다."""
        try:
            # 모듈의 create_map 함수 직접 호출 (필요시 인자 전달)
            map_file_path = map_creator.create_map() 
            return {"status": "success", "file_path": map_file_path}
        except HTTPException as e:
            logger.error(f"지도 생성 실패 (HTTPException): {e.status_code} - {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"지도 생성 중 예상치 못한 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"지도 생성 중 예상치 못한 서버 오류: {type(e).__name__}")
