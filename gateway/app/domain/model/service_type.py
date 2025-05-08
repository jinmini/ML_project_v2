from enum import Enum
import os

class ServiceType(str, Enum):
    CRIME = "crime"
    TITANIC = "titanic"
    NLP = "nlp"
    TF = "tf"
    CHAT = "chat"

# ✅ 환경 변수에서 서비스 URL 가져오기
CRIME_SERVICE_URL = os.getenv("CRIME_SERVICE_URL")
TITANIC_SERVICE_URL = os.getenv("TITANIC_SERVICE_URL")
NLP_SERVICE_URL = os.getenv("NLP_SERVICE_URL")
TF_SERVICE_URL = os.getenv("TF_SERVICE_URL")
CHAT_SERVICE_URL = os.getenv("CHAT_SERVICE_URL")

SERVICE_URLS = {
    ServiceType.CRIME: CRIME_SERVICE_URL,
    ServiceType.TITANIC: TITANIC_SERVICE_URL,
    ServiceType.NLP: NLP_SERVICE_URL,
    ServiceType.TF: TF_SERVICE_URL,
    ServiceType.CHAT: CHAT_SERVICE_URL,
}