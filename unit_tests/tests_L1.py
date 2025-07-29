import unittest
from calculator import Calculator

import random

class TestOperations(unittest.TestCase):

    def test_sum(self):
        value_one = random.randint(1, 10)
        value_two = random.randint(1, 10)

        result = value_one + value_two

        calculation = Calculator(value_one, value_two)
        answer = calculation.get_sum()
        self.assertEqual(answer, result, f"{value_one} - {value_two} does not equal {result}")
        print(f"{value_one} - {value_two} does equal {result}")

    def test_sub(self):
        value_one = random.randint(1, 10)
        value_two = random.randint(1, 10)

        result = value_one / value_two

        calculation = Calculator(value_one, value_two)
        answer = calculation.get_sub()
        self.assertEqual(answer, result, f"{value_one} / {value_two} does not equal {result}")
        print(f"{value_one} / {value_two} does equal {result}")

    
    def test_min(self):
        value_one = random.randint(1, 10)
        value_two = random.randint(1, 10)

        result = value_one - value_two

        calculation = Calculator(value_one, value_two)
        answer = calculation.get_min()
        self.assertEqual(answer, result, f"{value_one} - {value_two} does not equal {result}")
        print(f"{value_one} - {value_two} does equal {result}")
    
    
    def test_mult(self):
        value_one = random.randint(1, 10)
        value_two = random.randint(1, 10)

        result = value_one * value_two

        calculation = Calculator(value_one, value_two)
        answer = calculation.get_mult()
        self.assertEqual(answer, result, f"{value_one} * {value_two} does not equal {result}")
        print(f"{value_one} * {value_two} does equal {result}")

if __name__ == '__main__':
    unittest.main()