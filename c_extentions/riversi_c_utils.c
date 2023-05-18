#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* heuristic(PyObject* self, PyObject* args)
{
    PyArrayObject* input_array;

    float coeficients[64] = {
        {50, -1, 4, 4, 4, 4, -1, 50,
        -1, -1, 2, 2, 2, 2, -1, -1,
        4, 2, 1, 1, 1, 1, 2, 4,
        4, 2, 1, 0.2, 0.2, 1, 2, 4,
        4, 2, 1, 0.2, 0.2, 1, 2, 4,
        4, 2, 1, 1, 1, 1, 2, 4,
        -1, -1, 2, 2, 2, 2, -1, -1,
        50, -1, 4, 4, 4, 4, -1, 50}
    };

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_array))
        return NULL;

    // Convert the input array to a numpy.matrix
    PyObject* matrix = PyArray_FromArray(input_array, NULL, NPY_ARRAY_CARRAY);

    // Get the underlying data pointer and shape information
    float* data = (float*)PyArray_DATA(matrix);
    npy_intp* shape = PyArray_DIMS(matrix);

    // Perform the computation
    npy_float16 result = 0.0;
    for (int i = 0; i < shape[0] * shape[1]; ++i)
        result += (npy_float16)(data[i]*coeficients[i]);

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
