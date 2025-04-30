from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Dict, Optional

@dataclass
class TitanicSchema:
    train : object  
    test : object 
    context : str
    fname : str 
    id : str
    label : str

    @property
    def train(self) -> object:
        return self._train
    
    @train.setter
    def train(self, train):
        self._train = train
    
    @property
    def test(self) -> object:
        return self._test
    
    @test.setter
    def test(self, test):
        self._test = test
    
    @property
    def context(self) -> str:
        return self._context
    
    @context.setter
    def context(self, context):
        self._context = context
    
    @property
    def fname(self) -> str:
        return self._fname
    
    @fname.setter
    def fname(self, fname):
        self._fname = fname
    
    @property
    def id(self) -> str:
        return self._id
    
    @id.setter
    def id(self, id):
        self._id = id
    
    @property
    def label(self) -> str:
        return self._label
    
    @label.setter
    def label(self, label):
        self._label = label

# API 요청 모델
class TitanicRequest(BaseModel):
    pclass: Optional[int] = None
    sex: Optional[str] = None
    age: Optional[float] = None
    fare: Optional[float] = None
    embarked: Optional[str] = None
    
# API 응답 모델
class TitanicDataResponse(BaseModel):
    message: str = ""
    success: bool = True
    predictions: Optional[List[Dict[str, int]]] = None
    total_passengers: Optional[int] = None
    survived_count: Optional[int] = None
    death_count: Optional[int] = None
    