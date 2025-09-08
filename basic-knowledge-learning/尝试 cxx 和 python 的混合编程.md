## 尝试 C++ 和 Python 的混合编程

在实际开发中，我们经常会遇到需要将 C++ 的高性能与 Python 的易用性结合起来的场景。本文将介绍如何使用 pybind11 来实现 C++ 和 Python 的混合编程。

### 什么是 pybind11？

pybind11 是一个轻量级的库，用于将 C++ 代码绑定到 Python。它支持 C++11 及更高版本，并提供了简单而强大的 API 来创建 Python 模块。

### 安装 pybind11

可以使用 pip 来安装 pybind11：

```bash
pip install pybind11
```

或者从源码编译安装：

```bash
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build
cd build
cmake ..
make -j8
make install
```

### 一个简单的例子

下面我们来看一个简单的例子，展示如何使用 pybind11 将 C++ 函数暴露给 Python。

#### 步骤 1：创建 C++ 源文件

首先，创建一个名为 `example.cpp` 的文件，内容如下：

```cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 一个简单的 C++ 函数
double add(double a, double b) {
    return a + b;
}

// 一个带有默认参数的函数
double multiply(double a, double b = 2.0) {
    return a * b;
}

// 一个类
class Calculator {
public:
    Calculator() {}
    
    double add(double a, double b) {
        return a + b;
    }
    
    double subtract(double a, double b) {
        return a - b;
    }
};

// 绑定 Python 模块
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin";
    
    // 绑定函数
    m.def("add", &add, "A function which adds two numbers",
          py::arg("a"), py::arg("b"));
    
    m.def("multiply", &multiply, "A function which multiplies two numbers",
          py::arg("a"), py::arg("b") = 2.0);
    
    // 绑定类
    py::class_<Calculator>(m, "Calculator")
        .def(py::init<>())
        .def("add", &Calculator::add, "Add two numbers")
        .def("subtract", &Calculator::subtract, "Subtract two numbers");
}
```

#### 步骤 2：编译成 Python 模块

可以使用以下命令将 C++ 代码编译成 Python 模块：

```bash
g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`
```

或者使用 CMake 来构建：

```cmake
cmake_minimum_required(VERSION 3.4)
project(example)

# 寻找 Python 和 pybind11
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

# 添加模块
pybind11_add_module(example example.cpp)
```

#### 步骤 3：在 Python 中使用

编译完成后，可以在 Python 中直接导入并使用这个模块：

```python
import example

# 调用函数
print(example.add(1.0, 2.0))  # 输出: 3.0
print(example.multiply(3.0))  # 输出: 6.0
print(example.multiply(3.0, 4.0))  # 输出: 12.0

# 使用类
calc = example.Calculator()
print(calc.add(1.0, 2.0))  # 输出: 3.0
print(calc.subtract(5.0, 3.0))  # 输出: 2.0
```

### 更高级的功能

pybind11 还提供了许多更高级的功能，例如：

1. **支持 STL 容器**：可以自动转换 Python 列表、字典等与 C++ 的 vector、map 等容器
2. **异常处理**：可以在 C++ 和 Python 之间传递异常
3. **引用和指针**：支持引用和指针的传递
4. **继承和多态**：支持 C++ 类的继承和多态在 Python 中的使用
5. **回调函数**：支持 Python 函数作为 C++ 回调

### 示例：使用 STL 容器

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

namespace py = pybind11;

std::vector<int> filter_even(const std::vector<int>& numbers) {
    std::vector<int> result;
    for (int num : numbers) {
        if (num % 2 == 0) {
            result.push_back(num);
        }
    }
    return result;
}

PYBIND11_MODULE(example, m) {
    m.def("filter_even", &filter_even, "Filter even numbers");
}
```

在 Python 中使用：

```python
import example

numbers = [1, 2, 3, 4, 5, 6]
even_numbers = example.filter_even(numbers)
print(even_numbers)  # 输出: [2, 4, 6]
```

### 总结

跑通了
```bash
(base) abling@ABLing:~/Compilers_learning/MLsys_learning_2025summervacation/Jotang-summervacation-MLsysTour/basic-knowledge-learning/cxx-with-python$ mkdir build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python) \
         -DPython3_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))") \
         -DPython3_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
make -j
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
CMake Warning (dev) at CMakeLists.txt:5 (find_package):
  Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
  are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
  the cmake_policy command to set the policy and suppress this warning.

This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found PythonLibs: /usr/lib/x86_64-linux-gnu/libpython3.12.so (found version "3.12.3") 
CMake Warning (dev) at /usr/lib/cmake/pybind11/FindPythonLibsNew.cmake:98 (find_package):
  Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
  are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
  the cmake_policy command to set the policy and suppress this warning.

Call Stack (most recent call first):
  /usr/lib/cmake/pybind11/pybind11Tools.cmake:50 (find_package)
  /usr/lib/cmake/pybind11/pybind11Common.cmake:188 (include)
  /usr/lib/cmake/pybind11/pybind11Config.cmake:250 (include)
  CMakeLists.txt:6 (find_package)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found PythonInterp: /home/abling/miniconda3/bin/python (found suitable version "3.13.5", minimum required is "3.6") 
-- Found PythonLibs: /usr/lib/x86_64-linux-gnu/libpython3.12.so
-- Performing Test HAS_FLTO
-- Performing Test HAS_FLTO - Success
-- Found pybind11: /usr/include (found version "2.11.1")
-- Configuring done (13.2s)
-- Generating done (0.0s)
CMake Warning:
  Manually-specified variables were not used by the project:

    Python3_EXECUTABLE
    Python3_INCLUDE_DIR
    Python3_LIBRARY


-- Build files have been written to: /home/abling/Compilers_learning/MLsys_learning_2025summervacation/Jotang-summervacation-MLsysTour/basic-knowledge-learning/cxx-with-python/build
[ 50%] Building CXX object CMakeFiles/example.dir/example.cpp.o
[100%] Linking CXX shared module example.cpython-313-x86_64-linux-gnu.so
[100%] Built target example
(base) abling@ABLing:~/Compilers_learning/MLsys_learning_2025summervacation/Jotang-summervacation-MLsysTour/basic-knowledge-learning/cxx-with-python/build$ python ../trial.py 
3.0
6.0
12.0
3.0
2.0
```
使用 pybind11 可以很方便地将 C++ 代码绑定到 Python，让我们能够充分利用两种语言的优势：C++ 的高性能和 Python 的易用性。无论是为 Python 编写高性能的扩展模块，还是将现有的 C++ 代码库暴露给 Python，pybind11 都是一个非常好的选择。