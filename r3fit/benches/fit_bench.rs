use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;

use r3fit::Circle;

fn fit_circle(points: &[(f64, f64)], rng: &mut impl Rng) -> Circle {
    Circle::fit_with_rng(points, 1000, 0.1, rng).unwrap()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let points: Vec<(f64, f64)> = (0..1000).map(|_| (rng.random_range(0.0..100.0), rng.random_range(0.0..100.0))).collect();

    c.bench_function("fitting circle", |b| b.iter(|| fit_circle(black_box(&points), &mut rng)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);