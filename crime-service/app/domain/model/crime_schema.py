from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

@dataclass
class Dataset:
    cctv : object
    crime : object
    pop : object
    police : object

    @property
    def cctv(self) -> object:
        return self._cctv
    
    @cctv.setter
    def cctv(self, cctv):
        self._cctv = cctv
    
    @property
    def crime(self) -> object:
        return self._crime
    
    @crime.setter
    def crime(self, crime):
        self._crime = crime
    
    @property
    def pop(self) -> object:
        return self._pop
    
    @pop.setter
    def pop(self, pop):
        self._pop = pop
    
    @property
    def police(self) -> object:
        return self._police
    
    @police.setter
    def police(self, police):
        self._police = police


# API 요청 모델
class CrimeRequest(BaseModel):
    district: Optional[str] = None  # 지역구
    year: Optional[int] = None  # 년도
    crime_type: Optional[str] = None  # 범죄 유형 (살인, 강도, 강간, 절도, 폭력)
    
# API 응답 모델
class CrimeDataResponse(BaseModel):
    message: str = ""
    success: bool = True
    crime_data: Optional[Dict[str, Any]] = None
    total_districts: Optional[int] = None
    total_crimes: Optional[int] = None
    crime_rate: Optional[float] = None
    