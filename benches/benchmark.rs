use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use rand_pcg::Pcg64;
use voronoi::voronoi::*;

fn insert_many(n: usize) {
    const SEED: u64 = 333;
    let mut rng = Pcg64::seed_from_u64(SEED);
    let bounds = Bounds {
        min: -0.5,
        max: 0.5,
    };
    let mut delauney = Delauney::from_box([bounds, bounds]);
    for _ in 0..n {
        delauney.insert(Point {
            x: rng.gen_range(0.0..1.0),
            y: rng.gen_range(0.0..1.0),
        });
    }
}

fn from_scale(c: &mut Criterion) {
    static SCALE: usize = 1000;

    let mut group = c.benchmark_group("from_SCALE");

    for size in [SCALE, 2 * SCALE, 4 * SCALE, 8 * SCALE, 16 * SCALE].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                insert_many(size);
                black_box(())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, from_scale);
criterion_main!(benches);
