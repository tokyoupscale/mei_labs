import math
import numpy as np
from typing import List

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
    # радиус - единица чего либо
    def __init__(self, breed, age, name):
        self.breed = breed
        self.age = age
        self.name = name
    
    def tell_about(self) -> str:
        phrase = f"Кличка: {self.name}, возраст: {self.age}, порода: {self.breed}"
        return phrase

def main():
    print(f(x=0.5))
    print(forsum(0.1, 0.6, 0.1))
    print(whilesum(0.1, 0.6, 0.1))

    print(Dog("лабрадор", 5, 'челикс').tell_about())

if __name__ == "__main__":
    main()