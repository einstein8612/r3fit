use std::fmt;
use rand::Rng;
use thiserror::Error;


#[derive(Error, Debug)]
pub enum CircleError {
    #[error("Could not fit circle with {0} < 3 points")]
    NotEnoughPoints(i64),
    #[error("Points are collinear")]
    CollinearPoints,
}

#[derive(Clone, Copy, Debug)]
pub struct Circle {
    pub x: f64,
    pub y: f64,
    pub r: f64,
}

impl Circle {
    pub fn new(x: f64, y: f64, r: f64) -> Self {
        Circle { x, y, r }
    }

    pub fn distance_squared(&self, point: &(f64, f64)) -> f64 {
        let dx = self.x - point.0;
        let dy = self.y - point.1;
        dx * dx + dy * dy
    }

    pub fn count_inner_points(&self, points: &[(f64, f64)], threshold: f64) -> usize {
        points.iter().filter(|point| self.is_inner(point, threshold)).count()
    }

    pub fn is_inner(&self, point: &(f64, f64), threshold: f64) -> bool {
        let distance_2 = self.distance_squared(point);
        distance_2 > (self.r - threshold).powi(2) && distance_2 < (self.r + threshold).powi(2)
    }

    pub fn fit_with_rng(points: &[(f64, f64)], iter: usize, threshold: f64, rng: &mut impl Rng) -> Result<Circle, CircleError> {
        let n = points.len();
        if n < 3 {
            return Err(CircleError::NotEnoughPoints(n as i64));
        }

        let mut best_circle = Circle::new(0.0, 0.0, 0.0);
        let mut best_in_points = 0;

        for _ in 0..iter {
            let p0 = points[rng.random_range(0..n)];
            let p1 = points[rng.random_range(0..n)];
            let p2 = points[rng.random_range(0..n)];

            // If points are not collinear, fit the circle
            let fitted_circle = match fit_3_points(p0, p1, p2) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Check if the circle fits all points
            let in_points = fitted_circle.count_inner_points(points, threshold);
            if in_points > best_in_points {
                best_circle = fitted_circle;
                best_in_points = in_points;
            }

            // If we have a perfect fit, we can stop
            if best_in_points == n {
                break;
            }
        }

        Ok(best_circle)
    }

    pub fn fit(points: &[(f64, f64)], iter: usize, threshold: f64) -> Result<Circle, CircleError> {
        let mut rng = rand::rng();
        Self::fit_with_rng(points, iter, threshold, &mut rng)
    }
}

impl fmt::Display for Circle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Circle: center=({}, {}), radius={}", self.x, self.y, self.r)
    }
}

fn fit_3_points(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> Result<Circle, CircleError> {
    let d_0 = p0.0 * p0.0 + p0.1 * p0.1;
    let d_1 = p1.0 * p1.0 + p1.1 * p1.1;
    let d_2 = p2.0 * p2.0 + p2.1 * p2.1;
    let d_01 = d_0 - d_1;
    let d_02 = d_0 - d_2;
    
    let c = 2f64 * (p0.0 * (p1.1 - p2.1) + p1.0 * (p2.1 - p0.1) + p2.0 * (p0.1 - p1.1));
    if c == 0.0 {
        return Err(CircleError::CollinearPoints); // Points are collinear
    }
    let c_inv = 1.0 / c;

    let x_c = ((p0.1 - p2.1) * d_01 + (p1.1 - p0.1) * d_02) * c_inv;
    let y_c = ((p2.0 - p0.0) * d_01 + (p0.0 - p1.0) * d_02) * c_inv;
    let r = ((p0.0 - x_c).powi(2) + (p0.1 - y_c).powi(2)).sqrt();

    Ok(Circle::new(x_c, y_c, r))
}


#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    macro_rules! assert_float_eq {
        ($a:expr, $b:expr, $epsilon:expr) => {
            if ($a - $b).abs() > $epsilon {
                panic!(
                    "Assertion failed: {} and {} are not equal within epsilon {}",
                    $a, $b, $epsilon
                );
            }
        };
    }

    #[test]
    fn fit_perfect_circle_test() {
        let points = vec![
            (5.0, 8.0),
            (4.842915805643155, 9.243449435824274),
            (4.381533400219318, 10.408768370508577),
            (3.644843137107058, 11.422735529643443),
            (2.6791339748949827, 12.221639627510076),
            (1.5450849718747373, 12.755282581475768),
            (0.31395259764656763, 12.990133642141359),
            (-0.936906572928623, 12.911436253643444),
            (-2.1288964578253635, 12.524135262330098),
            (-3.1871199487434487, 11.852566213878946),
            (-4.045084971874736, 10.938926261462367),
            (-4.648882429441256, 9.84062276342339),
            (-4.960573506572389, 8.626666167821522),
            (-4.9605735065723895, 7.373333832178479),
            (-4.648882429441257, 6.15937723657661),
            (-4.045084971874739, 5.061073738537637),
            (-3.1871199487434474, 4.147433786121053),
            (-2.128896457825361, 3.475864737669901),
            (-0.9369065729286231, 3.0885637463565567),
            (0.31395259764656414, 3.009866357858642),
            (1.5450849718747361, 3.244717418524232),
            (2.6791339748949836, 3.7783603724899253),
            (3.644843137107056, 4.577264470356555),
            (4.381533400219316, 5.5912316294914195),
            (4.8429158056431545, 6.756550564175724),
        ];

        let c = Circle::fit(&points, 1000, 0.1).unwrap();

        // Circle should be centered at (0.0, 0.0) with radius 1.0
        assert_float_eq!(c.x, 0.0, 1e-5);
        assert_float_eq!(c.y, 8.0, 1e-5);
        assert_float_eq!(c.r, 5.0, 1e-5);
        assert!(c.count_inner_points(&points, 0.1) == points.len());
    }

    #[test]
    fn fit_best_circle_test() {
        let points = vec![
            (5.15, 4.97),
            (4.19, 5.94),
            (5.46, 3.46),
            (2.69, 6.11),
            (1.29, 5.29),
            (0.95, 3.67),
            (1.40, 2.00),
            (2.64, 1.10),
            (4.02, 1.01),
            (5.02, 2.00),
            (7.81, -0.25),
            (0.12, 7.77),
            (4.55, 8.60),
            (-1.00, 3.00),
            (6.66, 6.66)
        ];

        let c = Circle::fit(&points, 1000, 0.1).unwrap();

        // Circle should be centered at (3.0, 3.5) with radius 2.5

        assert_float_eq!(c.x, 3.0, 1.0);
        assert_float_eq!(c.y, 3.5, 1.0);
        assert_float_eq!(c.r, 2.5, 1.0);
        assert!(c.count_inner_points(&points, 0.1) > 4);
    }

    #[test]
    fn fit_best_circle_with_rng_test() {
        let points = vec![
            (5.15, 4.97),
            (4.19, 5.94),
            (5.46, 3.46),
            (2.69, 6.11),
            (1.29, 5.29),
            (0.95, 3.67),
            (1.40, 2.00),
            (2.64, 1.10),
            (4.02, 1.01),
            (5.02, 2.00),
            (7.81, -0.25),
            (0.12, 7.77),
            (4.55, 8.60),
            (-1.00, 3.00),
            (6.66, 6.66)
        ];

        let mut rng = StdRng::seed_from_u64(0);
        let c = Circle::fit_with_rng(&points, 1000, 0.1, &mut rng).unwrap();

        // Circle should be centered at (3.0, 3.5) with radius 2.5

        assert_float_eq!(c.x, 3.0, 0.5);
        assert_float_eq!(c.y, 3.5, 0.5);
        assert_float_eq!(c.r, 2.5, 0.5);
        assert_eq!(c.count_inner_points(&points, 0.1), 6);
    }

    #[test]
    fn thales_test() {
        let c = fit_3_points((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)).unwrap();

        // Circle should be centered at (0.5, 0.5) with radius sqrt(0.5)
        assert_float_eq!(c.x, 0.5, 1e-5);
        assert_float_eq!(c.y, 0.5, 1e-5);
        assert_float_eq!(c.r, 1.0/2f64.sqrt(), 1e-5);
    }

    #[test]
    fn arbitrary_circle_test() {
        let c = fit_3_points((1.0, 2.0), (5.0, 2.0), (3.0, 1.0)).unwrap();

        // Circle should be centered at (0.5, 0.5) with radius sqrt(0.5)
        assert_float_eq!(c.x, 3.0, 1e-5);
        assert_float_eq!(c.y, 3.5, 1e-5);
        assert_float_eq!(c.r, 2.5, 1e-5);
    }

    #[test]
    fn collinear_points_test() {
        let c = fit_3_points((0.0, 0.0), (1.0, 1.0), (2.0, 2.0));
        assert!(c.is_err()); // Should return None for collinear points, no circle can be formed
    }

    #[test]
    fn same_points_test() {
        let c = fit_3_points((1.0, 1.0), (1.0, 1.0), (1.0, 1.0));
        assert!(c.is_err()); // Should return None for identical points, as they are collinear
    }

    #[test]
    fn two_points_test() {
        let c = fit_3_points((1.0, 1.0), (3.0, 2.0), (1.0, 1.0));
        assert!(c.is_err()); // Should return None for two points and one identical point, as effectively only two points are provided and they are collinear by definition
    }

    #[test]
    fn not_enough_points_test() {
        let c = Circle::fit(&[(1.0, 1.0)], 1000, 0.1);
        assert!(c.is_err()); // Should return None for less than 3 points
    }

    #[test]
    fn display_test() {
        let c = Circle::new(1.0, 2.0, 3.0);
        assert_eq!(format!("{}", c), "Circle: center=(1, 2), radius=3");
    }

    proptest! {
        #[test]
        fn all_found_circles_are_correct(
            c_x in 0f64..100f64,
            c_y in 0f64..100f64,
            r2 in 10f64..100f64,
            off_p0_x in 0f64..3f64,
            off_p1_x in 0f64..3f64,
            off_p2_x in 0f64..3f64,
        ) {
            let p0_x = c_x + off_p0_x;
            let p1_x = c_x + off_p1_x;
            let p2_x = c_x + off_p2_x;
            let p0_y = c_y + (r2 - off_p0_x * off_p0_x).sqrt();
            let p1_y = c_y + (r2 - off_p1_x * off_p1_x).sqrt();
            let p2_y = c_y + (r2 - off_p2_x * off_p2_x).sqrt();
            let p0 = (p0_x, p0_y);
            let p1 = (p1_x, p1_y);
            let p2 = (p2_x, p2_y);
            let c_res = fit_3_points(p0, p1, p2);

            prop_assume!(c_res.is_ok());
            let c = c_res.unwrap();

            assert_float_eq!(c.x, c_x, 1e-5);
            assert_float_eq!(c.y, c_y, 1e-5);
            assert_float_eq!(c.r, r2.sqrt(), 1e-5);
        }
    }
}
