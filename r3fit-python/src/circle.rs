use pyo3::prelude::*;

use r3fit::Circle;

#[pyclass(name = "Circle")]
pub struct PyCircle {
   pub inner: Circle,
}

#[pymethods]
impl PyCircle {
    #[new]
    fn new(x: f64, y: f64, r: f64) -> Self {
        PyCircle {
            inner: Circle::new(x, y, r),
        }
    }

    #[getter]
    fn get_x(&self) -> PyResult<f64> {
        Ok(self.inner.x)
    }

    #[getter]
    fn get_y(&self) -> PyResult<f64> {
        Ok(self.inner.y)
    }

    #[getter]
    fn get_r(&self) -> PyResult<f64> {
        Ok(self.inner.r)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Circle(x: {}, y: {}, r: {})",
            self.inner.x, self.inner.y, self.inner.r
        ))
    }

    fn is_inside(&self, point: (f64, f64), threshold: f64) -> PyResult<bool> {
        Ok(self.inner.is_inner(&point, threshold))
    }

}