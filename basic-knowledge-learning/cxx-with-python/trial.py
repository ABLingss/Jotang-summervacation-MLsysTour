import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

import example

# 调用函数
print(example.add(1.0, 2.0))  # 输出: 3.0
print(example.multiply(3.0))  # 输出: 6.0
print(example.multiply(3.0, 4.0))  # 输出: 12.0

# 使用类
calc = example.Calculator()
print(calc.add(1.0, 2.0))  # 输出: 3.0
print(calc.subtract(5.0, 3.0))  # 输出: 2.0