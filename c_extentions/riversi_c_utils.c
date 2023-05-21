#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* heuristic(PyObject* self, PyObject* args)
{
    PyArrayObject* input_array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_array)) {
        return NULL;
    }

    if (PyArray_TYPE(input_array) != NPY_INT8) {
        PyErr_SetString(PyExc_TypeError, "Expected int8 matrix.");
        return NULL;
    }

    npy_intp* dims = PyArray_DIMS(input_array);
    if (dims[0] != 8 || dims[1] != 8) {
        PyErr_SetString(PyExc_ValueError, "Expected 8x8 matrix.");
        return NULL;
    }

    npy_int8* data = (npy_int8*)PyArray_DATA(input_array);
    npy_int16 sum = 0;
    for (npy_intp i = 0; i < dims[0] * dims[1]; i++) {
        sum += (npy_int16)data[i];
    }

    PyObject* result = PyFloat_FromDouble((double)sum);
    return result;
}

static PyMethodDef module_methods[] = {
    {"heuristic", heuristic, METH_VARARGS, "Calculates the heuristics of a riversi board."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "riversi_c_utils",          // Module name
    NULL,                 // Module docstring
    -1,                   // Size of per-interpreter state of the module
    module_methods        // Method table
};

PyMODINIT_FUNC PyInit_riversi_c_utils(void)
{
    import_array();  // Initialize numpy API

    return PyModule_Create(&moduledef);
}
