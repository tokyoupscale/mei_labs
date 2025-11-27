import math
from typing import List
import numpy as np

def f(x) -> float | None:    
    equation = 1.1*math.exp(x) + abs(math.cos(math.sqrt(math.pi*x))) - (4/9)
    return equation

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
    
    def whoami(self):
        name = self.name
        return print(f"эту собаку зовут {name}")
    
    def howold(self):
        age = self.age
        return print(f"этой собаке {age} лет")
    
    def whatkind(self):
        kind = self.kind
        return print(f"эта собака породы {kind}")

    
def main():
    print(f(x=0.5))
    print(forsum(0.1, 0.6, 0.1))
    print(whilesum(0.1, 0.6, 0.1))

    dog = Dog("Рекс", 5, "шпиц")
    dog.bark()
    dog.whoami()
    dog.howold()
    dog.whatkind()

if __name__ == "__main__":
    main()