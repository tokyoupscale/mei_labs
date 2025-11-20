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

class Circle:
    # радиус - единица чего либо
    def __init__(self, radius):
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius**2
    
    def length(self) -> float:
        return 2 * math.pi * self.radius
    
def main():
    print(f(x=0.5))
    print(forsum(0.1, 0.6, 0.1))
    print(whilesum(0.1, 0.6, 0.1))

    print(Circle(5).area())
    print(Circle(5).length())

if __name__ == "__main__":
    main()