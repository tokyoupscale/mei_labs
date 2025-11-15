import math

def f(x) -> float | 0:
    d1 = math.sqrt((1/5) + (math.exp(x))**(1/5))
    d2 = abs(math.log(x**2) - 1.3)
    
    if d2 == 0:
        return 0

    return d1/d2

class Circle:
    # радиус - единица чего либо
    def __init__(self, radius):
        self.radius = radius

    def area(self) -> float | 0:
        return math.pi * self.radius**2
    
    def length(self) -> float | 0:
        return 2 * math.pi * self.radius
    
def main():
    print(f(x=0.5))
    print(Circle(5).area())
    print(Circle(5).length())

if __name__ == "__main__":
    main()