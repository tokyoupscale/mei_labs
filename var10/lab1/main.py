import math
from typing import List
import numpy as np

def f(x) -> float:
    d1 = math.sqrt((1/5) + (math.exp(x))**(1/5))
    d2 = abs(math.log(x**2) - 1.3)
    
    if d2 == 0:
        return 0

    result = d1/d2
    return result

def forsum(start: float, end: float, step: float) -> List[float]:
    result = []
    for x in np.arange(start, end, step):
        sum = f(x)
        result.append(sum)
    return result

def whilesum(start: float, end: float, step: float) -> List[float]:
    result = []
    x = start
    while x <= end:
        sum = f(x)
        result.append(sum)
        x += step
    return result

class Dog:
    def __init__(self, name, age, kind):
        self.name = name
        self.age = age
        self.kind = kind

    def bark(self):
        return print("Bark!")
    
    def whoami(self, name: str):
        return print(f"эту собаку зовут {name}")
    
    def howold(self, age: int):
        return print(f"этой собаке {age} лет")
    
    def whatkind(self, kind :str):
        return print(f"эта собака породы {kind}")

    
def main():
    print(f(x=0.5))
    print(forsum(0.1, 0.6, 0.1))
    print(whilesum(0.1, 0.6, 0.1))

    dog = Dog("Рекс", 5, "шпиц")
    dog.bark()
    dog.whoami("Рекс")
    dog.howold(5)
    dog.whatkind("шпиц")

if __name__ == "__main__":
    main()