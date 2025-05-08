import json
from fastapi import APIRouter, FastAPI, Request, UploadFile, File, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import os
import logging
import sys
from typing import Optional 
from dotenv import load_dotenv
from pydantic import BaseModel
from app.domain.model.service_proxy_factory import ServiceProxyFactory
from contextlib import asynccontextmanager
from app.domain.model.service_type import ServiceType
import httpx

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("gateway_api")

# .env 파일 로드
load_dotenv()

# Swagger UI에서 tf/process 요청 본문 입력을 위한 Pydantic 모델
class TFProcessPayload(BaseModel):
    filename: str

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

# ✅ 서비스 프록시 팩토리 의존성 주입 함수
def get_proxy_factory(service: ServiceType) -> ServiceProxyFactory:
    return ServiceProxyFactory(service_type=service)

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
    request: Request,
    factory: ServiceProxyFactory = Depends(get_proxy_factory)
):
    response = await factory.request(
        method="GET",
        path=path,
        headers=request.headers.raw
    )
    return _handle_service_response(response)

# 파일 업로드 전용 엔드포인트
@gateway_router.post("/{service}/upload", summary="파일 업로드")
async def upload_file(
    service: ServiceType,
    request: Request,
    file: UploadFile = File(...),
    factory: ServiceProxyFactory = Depends(get_proxy_factory)
):
    logger.info(f"🌈Received file upload request for service: {service}")
    
    # 파일 데이터와 메타데이터를 multipart/form-data 형식으로 준비
    form_data = {"file": (file.filename, await file.read(), file.content_type)}
    
    # 추상화된 엔드포인트 경로 사용
    endpoint_path = f"{service.value}/upload"
    
    # 요청 전송 (파일 업로드를 위한 특별 처리)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{factory.base_url}/{endpoint_path}",
            files=form_data
        )
    
    return _handle_service_response(response)

# POST
@gateway_router.post("/{service}/{path:path}", summary="POST 프록시")
async def proxy_post(
    service: ServiceType,
    path: str,
    request: Request, # Request 객체는 헤더 및 쿼리 파라미터 접근을 위해 유지
    factory: ServiceProxyFactory = Depends(get_proxy_factory),
    # tf/process 경로에 대한 Swagger UI 요청 본문 활성화를 위한 파라미터
    tf_process_payload: Optional[TFProcessPayload] = Body(None)
):
    logger.info(f"🌈Received POST request for service: {service}, path: {path}")

    try:
        # 특수 처리: tf/process
        if service == ServiceType.TF and path == "process":
            filename = None
            # Pydantic 모델로 파싱된 payload가 있으면 사용
            if tf_process_payload:
                filename = tf_process_payload.filename
                logger.info(f"📦 tf/process: filename from Pydantic payload: {filename}")
            
            # Pydantic payload에서 filename을 얻지 못했다면 쿼리 파라미터에서 시도
            if not filename:
                filename = request.query_params.get("filename")
                if filename:
                    logger.info(f"📦 tf/process: filename from query parameters: {filename}")

            if not filename:
                logger.warning("❌ tf/process: filename 누락됨 (JSON body와 query param 모두 없음)")
                return JSONResponse(
                    status_code=422,
                    content={"detail": "필수 파라미터 'filename'이 누락되었습니다."}
                )

            # 요청 본문 재구성
            body_to_send = {"filename": filename}
            json_body = json.dumps(body_to_send).encode("utf-8")

            # 헤더 재구성 (원본 요청 헤더에서 Host 제외, Content-Type 및 Content-Length 재설정)
            headers_dict = {
                k.decode("utf-8").lower(): v.decode("utf-8") # 키를 소문자로 통일
                for k, v in request.headers.raw
                if k.decode("utf-8").lower() != "host" # Host 헤더 제외
            }
            headers_dict["content-type"] = "application/json" # 소문자로 설정
            headers_dict["content-length"] = str(len(json_body))

            logger.info(f"🔁 tf/process 요청 - 전달 데이터: {body_to_send}, 전달 헤더: {headers_dict}")
            response = await factory.request(
                method="POST",
                path=path, # path는 "process"
                headers=[(k.encode(), v.encode()) for k, v in headers_dict.items()],
                body=json_body
            )
        else:
            # 일반 POST 요청: 원본 요청 본문을 그대로 전달
            # 주의: tf_process_payload = Body(None)이 다른 경로의 요청 본문 처리에 영향을 미칠 수 있음
            # (예: tf_process_payload 파싱 시도 중 본문이 소비되어 아래 request.body()가 비어있을 수 있음)
            # FastAPI/Starlette의 request.body()는 일반적으로 캐시되므로 괜찮을 수 있으나, 주의 필요.
            body = await request.body()
            logger.debug(f"📦 일반 POST 요청 body: {body[:200]}...")
            logger.info(f"Forwarding to service. Headers: {request.headers.raw}")
            response = await factory.request(
                method="POST",
                path=path,
                headers=request.headers.raw, # 원본 헤더 사용
                body=body
            )

        return _handle_service_response(response)

    except Exception as e:
        logger.error(f"❗ POST 프록시 요청 처리 중 예외 발생: {type(e).__name__} - {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"요청 처리 중 오류가 발생했습니다: {str(e)}"}
        )

# PUT
@gateway_router.put("/{service}/{path:path}", summary="PUT 프록시")
async def proxy_put(
    service: ServiceType,
    path: str,
    request: Request,
    factory: ServiceProxyFactory = Depends(get_proxy_factory)
):
    logger.info(f"🌈Received PUT request for service: {service}, path: {path}")
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
async def proxy_delete(
    service: ServiceType,
    path: str,
    request: Request,
    factory: ServiceProxyFactory = Depends(get_proxy_factory)
):
    logger.info(f"🌈Received DELETE request for service: {service}, path: {path}")
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
async def proxy_patch(
    service: ServiceType,
    path: str,
    request: Request,
    factory: ServiceProxyFactory = Depends(get_proxy_factory)
):
    logger.info(f"🌈Received PATCH request for service: {service}, path: {path}")
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
app.include_router(gateway_router)

# ✅ 서버 실행
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True) 