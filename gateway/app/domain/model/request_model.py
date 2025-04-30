from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TitanicRequest(BaseModel):
    pclass: Optional[int] = None
    sex: Optional[str] = None
    age: Optional[float] = None
    fare: Optional[float] = None
    embarked: Optional[str] = None
    

