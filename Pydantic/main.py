from pydantic import BaseModel, Field ,field_validator as Validator
from typing import Optional
class User(BaseModel):
    id:int
    name:str
    email:str
    age:int
    isactive:bool
    @Validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v


user = User(
    id=1,
    name="Sabari",
    email="sabari@example.com",
    age=25,
    isactive=True
)

print(user)


class Main(BaseModel):
    name:str
    age:int

    def display(self):
        print(f"Name: {self.name}, Age: {self.age}")    
p=Main(name="Sabari",age=25)
p.display()


class Subclass(BaseModel):
    name:str=Field(min_length=3, max_length=50)
    age:int
    city:str
    country:str
    def display(self):
        print(f"Name:{self.name}, Age: {self.age}, City: {self.city}, Country: {self.country}")
a=Subclass(name="sabari",age=25, city="Chennai", country="India")
a.display()      


#logging configuration
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('example.log'),
        logging.StreamHandler()
    ],

)
logging.info("Logging from Pydantic example!")
logging.debug("This is a debug message")

class Dummy(BaseModel):
    name:str
    age:Optional[int] = None
    def display(self):
        print(f"Name: {self.name}, Age: {self.age if self.age is not None else 'Not provided'}")  
d=Dummy(name="Sabari",age=2)
d.display()        