from enum import Enum
import os

class ServiceType(str, Enum):
    CRIME = "crime"
    TITANIC = "titanic"

# ✅ 환경 변수에서 서비스 URL 가져오기
CRIME_SERVICE_URL = os.getenv("CRIME_SERVICE_URL")
TITANIC_SERVICE_URL = os.getenv("TITANIC_SERVICE_URL")

SERVICE_URLS = {
    ServiceType.CRIME: CRIME_SERVICE_URL,
    ServiceType.TITANIC: TITANIC_SERVICE_URL,
}