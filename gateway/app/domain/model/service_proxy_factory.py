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
        self.service_type = service_type
        logger.info(f"🔍 Service URL for {service_type.value}: {self.base_url}")
        
    async def request(
        self,
        method: str,
        path: str,
        headers: list[tuple[bytes, bytes]],
        body: Optional[bytes] = None
    ) -> httpx.Response:
        if path and self.service_type.value in path.lower():
            pass
        else:
            path = f"{self.service_type.value}/{path}"
        
        url = f"{self.base_url}/{path}"
        logger.info(f"🔍 Requesting URL: {url}")
        
        headers_dict = {}
        for k_bytes, v_bytes in headers:
            key = k_bytes.decode('utf-8').lower()
            value = v_bytes.decode('utf-8')
            if key != 'host':
                headers_dict[key] = value
        
        if 'content-type' not in headers_dict and body:
            headers_dict['Content-Type'] = 'application/json'
        if 'accept' not in headers_dict:
            headers_dict['Accept'] = 'application/json'
        
        logger.info(f"요청 정보: URL={url}, Method={method}, Headers={headers_dict}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                logger.info(f"HTTP 요청 시작: {method} {url}")
                if body:
                    logger.debug(f"Request body: {body.decode('utf-8') if isinstance(body, bytes) else body}")

                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers_dict,
                    content=body
                )
                logger.info(f"Response status: {response.status_code}")
                logger.debug(f"Request URL: {url}")
                logger.debug(f"Response body: {response.text}")

                return response
            except httpx.ConnectError as e:
                error_msg = f"서비스 연결 실패 (ConnectError): {str(e)}"
                logger.error(error_msg)
                logger.error(f"상세 에러 트레이스백:\n{traceback.format_exc()}")
                raise HTTPException(status_code=503, detail=error_msg)
            except httpx.TimeoutException as e:
                error_msg = f"서비스 타임아웃 (TimeoutException): {str(e)}"
                logger.error(error_msg)
                logger.error(f"상세 에러 트레이스백:\n{traceback.format_exc()}")
                raise HTTPException(status_code=504, detail=error_msg)
            except httpx.RequestError as e:
                error_msg = f"요청 오류 (RequestError): {str(e)}"
                logger.error(error_msg)
                logger.error(f"상세 에러 트레이스백:\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=error_msg)
            except Exception as e:
                error_msg = f"프록시 처리 중 예상치 못한 오류 발생: {str(e)}"
                error_traceback = traceback.format_exc()
                logger.error(error_msg)
                logger.error(f"상세 에러 트레이스백:\n{error_traceback}")
                raise HTTPException(status_code=500, detail=error_msg)