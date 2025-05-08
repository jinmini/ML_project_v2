import logging
import sys

# 로거 인스턴스 생성
logger = logging.getLogger("nlp_service")
logger.setLevel(logging.INFO)  # 로그 레벨 설정 (INFO 이상만 기록)

# 포매터 생성
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 스트림 핸들러 생성 (콘솔 출력)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

# 로거에 핸들러 추가
logger.addHandler(stream_handler)

# 다른 모듈에서 이 logger 객체를 임포트하여 사용합니다.
# 예: from app.utils.logger import logger 