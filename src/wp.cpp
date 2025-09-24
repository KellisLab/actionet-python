#include <nanobind/nanobind.h>

int add(int a, int b) { return a + b; }

NB_MODULE(test, m) {
    m.def("add", &add);
}
