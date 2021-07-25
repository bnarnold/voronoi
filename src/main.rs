struct Point {
    x: f64,
    y: f64,
}

enum TriangleStatus {
    Alive,
    Dead,
}

struct Triangle {
    vertices: [usize; 3],
    neighbors: [usize; 3],
    alive: TriangleStatus,
}

struct Voronoi {
    points: Vec<Point>,
    triangles: Vec<Triangle>,
}

impl Voronoi {
    fn from_box(bounds: [f64; 4]) -> Voronoi {
        let bottom = Triangle {
            vertices: [0, 1, 2],
            neighbors: [0, 1, 0],
            alive: TriangleStatus::Alive,
        };
        let top = Triangle {
            vertices: [0, 2, 3],
            neighbors: [1, 1, 0],
            alive: TriangleStatus::Alive,
        };
        let x_inds = [0, 1, 1, 0];
        let y_inds = [2, 2, 3, 3];
        let points = panic!();
        Voronoi {
            points,
            triangles: vec![bottom, top],
        }
    }
}

impl Point {
    fn dist(self, other: Point) -> f64 {
        ((other.x - self.x).powi(2) + (other.y - self.y).powi(2)).sqrt()
    }
}

fn main() {
    println!("Hello, world!");
}
