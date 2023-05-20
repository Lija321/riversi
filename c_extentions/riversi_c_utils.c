#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* heuristic(PyObject* self, PyObject* args)
{
    PyArrayObject* matrix;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &matrix)) {
        return NULL;
    }

    if (PyArray_NDIM(matrix) != 2 || PyArray_TYPE(matrix) != NPY_INT8 ||
        PyArray_DIM(matrix, 0) != 8 || PyArray_DIM(matrix, 1) != 8) {
        PyErr_SetString(PyExc_TypeError, "Expected an 8x8 int8 matrix");
        return NULL;
    }

    // Get a pointer to the matrix data
    int8_t* data = (int8_t*)PyArray_DATA(matrix);

    // Calculate the heuristic value
    npy_float16 result = 0.0;
    for (int i = 0; i < 64; i++) {
        result += (npy_float16)data[i];
    }

    // Clean up
    Py_DECREF(matrix);

    // Return the result as a numpy.float16
    return PyFloat_FromDouble((double)result);
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
