class Calculator:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def get_sum(self):
        return self.a + self.b
    
    def get_sub(self):
        return self.a - self.b
    
    def get_div(self):
        return self.a / self.b
    
    def get_mult(self):
        return self.a * self.b

myCalc = Calculator(a=2, b=5) 