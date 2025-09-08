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