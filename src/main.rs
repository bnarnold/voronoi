#[derive(Copy, Clone, Debug)]
struct Point {
    x: f64,
    y: f64,
}

enum TriangleStatus {
    Alive {
        neighbors: [(usize, TriangleIndex); 3],
    },
    Dead {
        descendants: Vec<usize>,
    },
}

struct Bounds {
    min: f64,
    max: f64,
}

struct Triangle {
    vertices: [usize; 3],
    alive: TriangleStatus,
}

struct Voronoi {
    points: Vec<Point>,
    triangles: Vec<Triangle>,
}
#[derive(Copy, Clone, Debug)]
enum TriangleIndex {
    First,
    Second,
    Third,
}

impl TriangleIndex {
    fn next(self) -> TriangleIndex {
        match self {
            TriangleIndex::First => TriangleIndex::Second,
            TriangleIndex::Second => TriangleIndex::Third,
            TriangleIndex::Third => TriangleIndex::First,
        }
    }

    fn all() -> impl Iterator<Item = &'static TriangleIndex> {
        [
            TriangleIndex::First,
            TriangleIndex::Second,
            TriangleIndex::Third,
        ]
        .iter()
    }
    fn lookup<T: Copy>(self, array: &[T; 3]) -> T {
        match self {
            TriangleIndex::First => array[0],
            TriangleIndex::Second => array[1],
            TriangleIndex::Third => array[2],
        }
    }
}

impl Voronoi {
    fn from_box(bounds: [Bounds; 2]) -> Voronoi {
        let bottom = Triangle {
            vertices: [0, 1, 2],
            alive: TriangleStatus::Alive {
                neighbors: [
                    (0, TriangleIndex::First),
                    (1, TriangleIndex::Third),
                    (0, TriangleIndex::Third),
                ],
            },
        };
        let top = Triangle {
            vertices: [0, 2, 3],
            alive: TriangleStatus::Alive {
                neighbors: [
                    (1, TriangleIndex::First),
                    (1, TriangleIndex::Second),
                    (0, TriangleIndex::Second),
                ],
            },
        };
        let xs = [bounds[0].min, bounds[1].max, bounds[1].max, bounds[0].min];
        let ys = [bounds[0].min, bounds[0].max, bounds[1].max, bounds[1].min];
        let points = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| Point { x, y })
            .collect();
        Voronoi {
            points,
            triangles: vec![bottom, top],
        }
    }
    fn contains(&self, p: Point) -> Option<usize> {
        let mut i: usize = 0;
        if !(self.triangle_contains(i, p)) {
            if self.triangle_contains(1, p) {
                i = 1;
            } else {
                return None;
            }
        }
        loop {
            match &self.triangles[i].alive {
                TriangleStatus::Alive { .. } => return Some(i),
                TriangleStatus::Dead { descendants } => {
                    for &j in descendants {
                        if self.triangle_contains(j, p) {
                            i = j;
                            break;
                        }
                    }
                }
            }
        }
    }
    fn triangle_contains(&self, i: usize, p: Point) -> bool {
        TriangleIndex::all().all(|&n| self.edge_contains(i, n, p))
    }
    fn edge_contains(&self, i: usize, n: TriangleIndex, p: Point) -> bool {
        let tri = &self.triangles[i];
        let left = n.lookup(&tri.vertices);
        let right = n.next().lookup(&tri.vertices);
        self.points[left].ccw(self.points[right], p)
    }
    fn split_triangle(&mut self, p: Point) -> Vec<(usize, TriangleIndex)> {
        match self.contains(p) {
            None => vec![],
            Some(i) => todo!(),
        }
    }
    fn flip_triangle(&mut self, i: usize, n: TriangleIndex) -> Vec<(usize, TriangleIndex)> {
        todo!()
    }
    fn insert(&mut self, p: Point) {
        let mut queue = self.split_triangle(p);
        match queue.pop() {
            None => return,
            Some((i, n)) => queue.append(&mut self.flip_triangle(i, n)),
        }
    }
}

impl Point {
    fn dist(self, other: Point) -> f64 {
        ((other.x - self.x).powi(2) + (other.y - self.y).powi(2)).sqrt()
    }
    fn ccw(self, p1: Point, p2: Point) -> bool {
        return (p2.x - self.x) * (p1.y - self.y) < (p2.y - self.y) * (p1.x - self.x);
    }
}

fn main() {
    println!("Compiles!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccw_orientation() {
        let origin = Point { x: 0.0, y: 0.0 };
        let px = Point { x: 1.0, y: 0.0 };
        let py = Point { x: 0.0, y: 1.0 };
        assert_eq!(origin.ccw(px, py), true)
    }
}
