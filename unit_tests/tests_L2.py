import unittest
from calculator import Calculator

class TestCalculation(unittest.TestCase):
    def setUp(self):
        self.operator = Calculator(a=8, b=2)
    
    def test_sum(self):
        self.assertEqual(self.operator.get_sum(), 10, 'The sum is wrong')

    def test_sub(self):
        self.assertEqual(self.operator.get_sub(), 6, 'The sum is wrong')

    def tearDown(self):
        print('All test ran. Goodbye')

if __name__ == '__main__':
    unittest.main()

    
