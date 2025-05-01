import pytest
from httpx import AsyncClient
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():

    return {"message": "테스트 파일입니다"}

@pytest.mark.asyncio
async def test_read_root(async_client: AsyncClient): 

    response = await async_client.get("/test")
    # 응답 상태 코드가 200인지 확인합니다.
    assert response.status_code == 200
    # 응답 JSON 본문이 예상된 메시지와 일치하는지 확인합니다.
    assert response.json() == {"message": "테스트 파일입니다🤩☺️"}


