# #!/usr/bin/env python

# import asyncio
# import httpx
# import sys
# import logging

# # 로깅 설정
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler(sys.stdout)]
# )
# logger = logging.getLogger("connection_test")

# TITANIC_SERVICE_URL = "http://titanic:9001"

# async def test_titanic_connection():
#     """Titanic 서비스 연결 테스트"""
#     logger.info(f"Titanic 서비스 연결 테스트 시작: {TITANIC_SERVICE_URL}")
    
#     # 1. 기본 헬스체크 엔드포인트 테스트
#     health_url = f"{TITANIC_SERVICE_URL}/titanic/health"
#     logger.info(f"헬스체크 URL: {health_url}")
    
#     try:
#         logger.info("타임아웃 30초로 설정하여 요청 시작...")
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             response = await client.get(health_url)
#             logger.info(f"응답 상태: {response.status_code}")
#             logger.info(f"응답 내용: {response.text}")
#             return True
#     except httpx.ConnectError as e:
#         logger.error(f"연결 오류: {str(e)}")
#         return False
#     except httpx.TimeoutException as e:
#         logger.error(f"타임아웃: {str(e)}")
#         return False
#     except Exception as e:
#         logger.error(f"예외 발생: {str(e)}")
#         return False

# async def test_network_connectivity():
#     """네트워크 연결성 테스트"""
#     logger.info("네트워크 연결성 테스트 시작")
    
#     tests = [
#         # Titanic 서비스 테스트
#         test_titanic_connection(),
#     ]
    
#     results = await asyncio.gather(*tests, return_exceptions=True)
#     logger.info(f"테스트 결과: {results}")
    
#     if all(results):
#         logger.info("모든 서비스 연결 성공! 🎉")
#     else:
#         logger.error("일부 서비스 연결 실패! 😞")

# if __name__ == "__main__":
#     logger.info("=== 서비스 연결 테스트 시작 ===")
#     asyncio.run(test_network_connectivity())
#     logger.info("=== 서비스 연결 테스트 완료 ===") 