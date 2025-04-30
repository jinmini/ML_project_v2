from typing import Optional
from fastapi import HTTPException
import httpx
import traceback
import logging
from app.domain.model.service_type import SERVICE_URLS, ServiceType

logger = logging.getLogger("gateway_api")

class ServiceProxyFactory:
    def __init__(self, service_type: ServiceType):
        self.base_url = SERVICE_URLS[service_type]
        print(f"🔍 Service URL: {self.base_url}")
    async def request(
        self,
        method: str,
        path: str,
        headers: list[tuple[bytes, bytes]],
        body: Optional[bytes] = None
    ) -> httpx.Response:
        # titanic-service의 라우터는 prefix="/titanic"에 등록되어 있으므로
        # 경로를 적절하게 조정합니다
        if path and ServiceType.TITANIC.value in path.lower():
            # 경로에 이미 'titanic'이 포함되어 있으면 그대로 사용
            pass
        elif path and ServiceType.CRIME.value in path.lower():
            # 경로에 이미 'crime'이 포함되어 있으면 그대로 사용
            pass
        else:
            # 서비스 타입에 따라 앞에 적절한 경로 추가
            if self.base_url == SERVICE_URLS[ServiceType.TITANIC]:
                path = f"titanic/{path}"
            elif self.base_url == SERVICE_URLS[ServiceType.CRIME]:
                path = f"crime/{path}"
        
        url = f"{self.base_url}/{path}"
        print(f"🔍 Requesting URL: {url}")
        # 헤더 설정
        headers_dict = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        logger.info(f"요청 정보: URL={url}, Method={method}, Headers={headers_dict}")
        
        # 명시적인 타임아웃 설정 추가
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                logger.info(f"HTTP 요청 시작: {method} {url}")
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers_dict,
                    content=body
                )
                print(f"Response status: {response.status_code}")
                print(f"Request URL: {url}")
                print(f"Request body: {body}")
                
                # 응답 본문을 로그로 추가
                print(f"Response body: {response.text}")
                
                return response
            except httpx.ConnectError as e:
                error_msg = f"서비스 연결 실패 (ConnectError): {str(e)}"
                logger.error(error_msg)
                logger.error(f"상세 에러 트레이스백:\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=error_msg)
            except httpx.TimeoutException as e:
                error_msg = f"서비스 타임아웃 (TimeoutException): {str(e)}"
                logger.error(error_msg)
                logger.error(f"상세 에러 트레이스백:\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=error_msg)
            except httpx.RequestError as e:
                error_msg = f"요청 오류 (RequestError): {str(e)}"
                logger.error(error_msg)
                logger.error(f"상세 에러 트레이스백:\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=error_msg)
            except Exception as e:
                print(f"Request failed: {str(e)}")
                # 전체 트레이스백 로깅 추가
                error_traceback = traceback.format_exc()
                print(f"상세 에러 트레이스백:\n{error_traceback}")
                logger.error(f"일반 예외 발생: {str(e)}")
                logger.error(f"상세 에러 트레이스백:\n{error_traceback}")
                raise HTTPException(status_code=500, detail=str(e))