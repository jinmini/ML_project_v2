import json
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import os
import logging
import sys
from dotenv import load_dotenv
from app.domain.model.service_proxy_factory import ServiceProxyFactory
from contextlib import asynccontextmanager
from app.domain.model.service_type import ServiceType

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("gateway_api")

# .env 파일 로드
load_dotenv()

# ✅ 애플리케이션 시작 시 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Gateway API 서비스 시작")
    yield
    logger.info("🛑 Gateway API 서비스 종료")


# ✅ FastAPI 앱 생성
app = FastAPI(
    title="Gateway API",
    description="Gateway API for jinmini.com",
    version="0.1.0",
    lifespan=lifespan
)

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 메인 라우터 생성
gateway_router = APIRouter(prefix="/ai", tags=["AI API"])

# ✅ 헬스 체크 엔드포인트 추가
@gateway_router.get("/health", summary="테스트 엔드포인트")
async def health_check():
    return {"status": "healthy!"}

# --- Helper Function for Response Handling ---
def _handle_service_response(response) -> Response:
    """서비스 응답을 Content-Type에 따라 적절한 FastAPI 응답 객체로 변환"""
    content_type = response.headers.get("content-type", "")
    # JSON 응답이면 JSON으로 처리
    if "application/json" in content_type:
        try:
            # 응답 내용을 로그로 남기기 (DEBUG 레벨 추천)
            response_text = response.text
            logger.debug(f"서비스 응답 내용 (JSON): {response_text}")
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except json.JSONDecodeError:
            # 응답이 JSON 형식이 아닌 경우 (Content-Type이 잘못 지정된 경우 대비)
            logger.error(f"JSON 디코딩 오류 발생 (Content-Type: {content_type}): {response.text}")
            # 원시 응답으로 처리
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=content_type # 원본 Content-Type 유지
            )
    # HTML, 이미지 등 비JSON 응답은 그대로 반환
    else:
        logger.debug(f"서비스 응답 내용 (Non-JSON, Content-Type: {content_type}): {response.content[:200]}...") # 내용 일부만 로깅
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=content_type
        )

# --- Proxy Endpoints ---

# GET
@gateway_router.get("/{service}/{path:path}", summary="GET 프록시")
async def proxy_get(
    service: ServiceType,
    path: str,
    request: Request
):
    factory = ServiceProxyFactory(service_type=service)
    response = await factory.request(
        method="GET",
        path=path,
        headers=request.headers.raw
    )
    return _handle_service_response(response)

# POST
@gateway_router.post("/{service}/{path:path}", summary="POST 프록시")
async def proxy_post(
    service: ServiceType,
    path: str,
    request: Request
):
    logger.info(f"🌈Received POST request for service: {service}, path: {path}")
    factory = ServiceProxyFactory(service_type=service)
    body = await request.body()
    logger.debug(f"Raw request body: {body[:200]}...")

    response = await factory.request(
        method="POST",
        path=path,
        headers=request.headers.raw,
        body=body
    )
    return _handle_service_response(response)

# PUT
@gateway_router.put("/{service}/{path:path}", summary="PUT 프록시")
async def proxy_put(service: ServiceType, path: str, request: Request):
    logger.info(f"🌈Received PUT request for service: {service}, path: {path}")
    factory = ServiceProxyFactory(service_type=service)
    body = await request.body()
    logger.debug(f"Raw request body: {body[:200]}...")
    response = await factory.request(
        method="PUT",
        path=path,
        headers=request.headers.raw,
        body=body
    )
    return _handle_service_response(response)

# DELETE
@gateway_router.delete("/{service}/{path:path}", summary="DELETE 프록시")
async def proxy_delete(service: ServiceType, path: str, request: Request):
    logger.info(f"🌈Received DELETE request for service: {service}, path: {path}")
    factory = ServiceProxyFactory(service_type=service)
    body = await request.body()
    logger.debug(f"Raw request body: {body[:200]}...")
    response = await factory.request(
        method="DELETE",
        path=path,
        headers=request.headers.raw,
        body=body
    )
    return _handle_service_response(response)

# PATCH
@gateway_router.patch("/{service}/{path:path}", summary="PATCH 프록시")
async def proxy_patch(service: ServiceType, path: str, request: Request):
    logger.info(f"🌈Received PATCH request for service: {service}, path: {path}")
    factory = ServiceProxyFactory(service_type=service)
    body = await request.body()
    logger.debug(f"Raw request body: {body[:200]}...")
    response = await factory.request(
        method="PATCH",
        path=path,
        headers=request.headers.raw,
        body=body
    )
    return _handle_service_response(response)

# ✅ 라우터 등록
app.include_router(gateway_router, tags=["Gateway API"])

# ✅ 서버 실행
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True) 