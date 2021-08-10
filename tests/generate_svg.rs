extern crate voronoi;
use rand::prelude::*;
use voronoi::voronoi::*;

fn random_delauney(n: usize) -> Delauney {
    const SEED: u64 = 33;
    let mut rng = StdRng::seed_from_u64(SEED);
    let bounds = Bounds {
        min: -200.0,
        max: 1200.0,
    };
    let mut delauney = Delauney::from_box([bounds, bounds]);
    for _ in 0..n {
        delauney.insert(Point {
            x: rng.gen_range(0.0..1000.0),
            y: rng.gen_range(0.0..1000.0),
        });
    }
    delauney
}

#[test]
fn test_uniform_dist() {
    const N: usize = 20000;
    let delauney = random_delauney(N);
    delauney.save_svg(format!("target/test_output/uniform{}.svg", N));
}

#[test]
fn test_ellipse() {
    const N: usize = 50;
    let bounds = Bounds {
        min: -200.0,
        max: 1200.0,
    };
    let mut delauney = Delauney::from_box([bounds, bounds]);
    for i in 0..N {
        let t = 2.0 * std::f64::consts::PI * (i as f64) / (N as f64);
        delauney.insert(Point {
            x: 500.0 + 100.0 * t.cos(),
            y: 500.0 + 250.0 * t.sin(),
        });
    }
    delauney.save_svg(format!("target/test_output/ellipse{}.svg", N));
}
