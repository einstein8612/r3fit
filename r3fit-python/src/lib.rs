mod circle;
use circle::PyCircle;

use pyo3::prelude::*;
use numpy::PyReadonlyArray2;

use std::slice;

use ::r3fit::Circle;

#[pyfunction]
///
/// Fits a circle to a set of points using the RANSAC algorithm
/// as implemented in the r3fit crate.
///
/// ### Arguments
/// * `data` - A (n,2) numpy array of type f64 or a list of tuples of (x, y) coordinates.
/// * `iter` - The number of iterations to run the RANSAC algorithm.
/// * `threshold` - The threshold for determining whether a point is inliers.
///
/// ### Returns
/// * A `Circle` object containing the center and radius of the fitted circle.
///
fn fit<'py>(py: Python<'py>, data: Py<PyAny>, iter: usize, threshold: f64) -> PyResult<PyCircle> {
    // Check if the input is a numpy array
    if let Ok(array) = data.extract::<PyReadonlyArray2<f64>>(py) {
        return fit_numpy(array, iter, threshold);
    };

    // Check if the input is a list of tuples
    if let Ok(list) = data.extract::<Vec<(f64,f64)>>(py) {
        return fit_list(list, iter, threshold);
    };

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Input must be a (n,2) numpy array of type f64 or a list of tuples of (x, y) coordinates",
    ))
}

///
/// Fits a circle to a set of points using the RANSAC algorithm
/// as implemented in the r3fit crate.
///
/// ### Arguments
/// * `array` - A (n,2) numpy array of type f64.
/// * `iter` - The number of iterations to run the RANSAC algorithm.
/// * `threshold` - The threshold for determining whether a point is inliers.
///
/// ### Returns
/// * A `Circle` object containing the center and radius of the fitted circle.
///
fn fit_numpy(array: PyReadonlyArray2<f64>, iter: usize, threshold: f64) -> PyResult<PyCircle> {
    let array = array.as_array();  // returns ndarray::ArrayViewD<f64>
    let len = array.len();

    if array.shape()[1] != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Array must be of shape (n, 2)",
        ));
    }

    let ptr = array.as_ptr();
    let slice = unsafe {
        slice::from_raw_parts(ptr, len)
            .chunks(2) // Group the raw slice into chunks of two elements
            .map(|chunk| (chunk[0], chunk[1])) // Create tuples
            .collect::<Vec<(f64, f64)>>()
    };

    let circle = match Circle::fit(&slice, iter, threshold) {
        Ok(circle) => circle,
        Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())),
    };

    Ok(PyCircle {
        inner: circle,
    })
}


///
/// Fits a circle to a set of points using the RANSAC algorithm
/// as implemented in the r3fit crate.
///
/// ### Arguments
/// * `array` - A (n,2) numpy array of type f64.
/// * `iter` - The number of iterations to run the RANSAC algorithm.
/// * `threshold` - The threshold for determining whether a point is inliers.
///
/// ### Returns
/// * A `Circle` object containing the center and radius of the fitted circle.
///
fn fit_list(list: Vec<(f64, f64)>, iter: usize, threshold: f64) -> PyResult<PyCircle> {
    let circle = match Circle::fit(&list, iter, threshold) {
        Ok(circle) => circle,
        Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())),
    };

    Ok(PyCircle {
        inner: circle,
    })
}



/// A Python module implemented in Rust.
#[pymodule]
fn r3fit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    Ok(())
}
