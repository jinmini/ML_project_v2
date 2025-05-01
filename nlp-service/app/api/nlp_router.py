from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from app.domain.service.samsung_report import SamsungReportAnalyzer
from app.utils.logger import logger

router = APIRouter()

@router.get(
    "/wordcloud",
    response_class=FileResponse, # 응답이 파일임을 명시
    summary="삼성 리포트 워드클라우드 생성",
    description="제공된 삼성 리포트 텍스트 파일을 분석하여 워드클라우드 이미지를 생성하고 반환합니다.",
    responses={
        200: {
            "description": "성공적으로 워드클라우드 이미지 생성 및 반환",
            "content": {"image/png": {}}
        },
        500: {
            "description": "내부 서버 오류 (예: 파일 처리 실패, 워드클라우드 생성 실패)"
        }
    }
)
async def generate_wordcloud():
    """
    Samsung Report 텍스트를 분석하여 워드클라우드 PNG 이미지를 반환합니다.
    """
    logger.info("워드클라우드 생성 요청 수신")
    try:
        # 분석기 인스턴스 생성 (기본 경로 사용)
        analyzer = SamsungReportAnalyzer()
        # 분석 프로세스 실행 및 결과 이미지 경로 받기
        # 임시 파일 이름 지정 (요청마다 고유하게 하려면 UUID 등을 사용할 수 있음)
        output_filename = "samsung_report_wc_result.png"
        image_path = analyzer.process(output_image_path=output_filename)

        if image_path:
            logger.info(f"분석 완료, 이미지 파일 반환: {image_path}")
            # FileResponse를 사용하여 생성된 이미지 파일 반환
            return FileResponse(path=image_path, media_type='image/png', filename=output_filename)
        else:
            logger.error("워드클라우드 생성 실패 또는 결과 없음")
            # 분석 과정에서 오류 발생 시 500 에러 반환
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="워드클라우드 생성에 실패했습니다."
            )

    except Exception as e:
        logger.exception(f"API 처리 중 예외 발생: {e}")
        # 그 외 예외 발생 시 500 에러 반환
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"워드클라우드 생성 중 오류 발생: {str(e)}"
        )
