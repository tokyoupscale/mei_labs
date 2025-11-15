import math

def f(x) -> float | None:    

    return 1.1*math.exp(x) + abs(math.cos(math.sqrt(math.pi*x))) - (4/9)

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
    print(Dog("лабрадор", 5, 'челикс').tell_about())
    pass

if __name__ == "__main__":
    main()