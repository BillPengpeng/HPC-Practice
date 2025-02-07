#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
  for (size_t start = 0; start < m; start += batch)
  {
    size_t end = (start + batch < m) ? start + batch : m;
    const float *X_batch = X + start * n;
    const unsigned char *y_batch = y + start;
    size_t nums = end - start;
    std::vector<std::vector<float>> pred(nums, std::vector<float>(k, 0));
    std::vector<std::vector<float>> grad(n, std::vector<float>(k, 0));
    for (size_t i = 0; i < nums; i++)
    {
      for (size_t j = 0; j < n; j++)
      {
        for (size_t t = 0; t < k; t++)
        {
          float val = X_batch[i * n + j] * theta[j * k + t];
          pred[i][t] += val;
        }
      }
      float sum = 0;
      for (size_t t = 0; t < k; t++)
      {
        sum += exp(pred[i][t]);
      }
      for (size_t t = 0; t < k; t++)
      {
        pred[i][t] = exp(pred[i][t]) / sum;
      }
    }
    for (size_t i = 0; i < n; i++)
    {
      for (size_t j = 0; j < nums; j++)
      {
        for (size_t t = 0; t < k; t++)
        {
          if (t == y_batch[j])
            grad[i][t] += X_batch[j * n + i] * (pred[j][t] - 1);
          else
            grad[i][t] += X_batch[j * n + i] * pred[j][t];
        }
      }
    }
    for (size_t i = 0; i < n; i++)
    {
      for (size_t t = 0; t < k; t++)
      {
        theta[i * k + t] -= (lr / nums * grad[i][t]);
      }
    }
  }
    
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
