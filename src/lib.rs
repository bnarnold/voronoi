pub mod voronoi {
    use std::iter;
    #[derive(Copy, Clone, Debug)]
    pub struct Point {
        x: f64,
        y: f64,
    }

    #[derive(Clone, Debug)]
    enum TriangleStatus {
        Alive {
            neighbors: [(usize, TriangleIndex); 3],
        },
        Dead {
            descendants: Vec<usize>,
        },
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Bounds {
        min: f64,
        max: f64,
    }

    #[derive(Clone, Debug)]
    struct Triangle {
        vertices: [usize; 3],
        alive: TriangleStatus,
    }

    #[derive(Clone, Debug)]
    pub struct Voronoi {
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

        fn prev(self) -> TriangleIndex {
            match self {
                TriangleIndex::First => TriangleIndex::Third,
                TriangleIndex::Second => TriangleIndex::First,
                TriangleIndex::Third => TriangleIndex::Second,
            }
        }
        fn all() -> impl Iterator<Item = TriangleIndex> {
            iter::once(TriangleIndex::First)
                .chain(iter::once(TriangleIndex::Second))
                .chain(iter::once(TriangleIndex::Third))
        }
        fn lookup<T>(self, array: [T; 3]) -> T
        where
            T: Copy,
        {
            match self {
                TriangleIndex::First => array[0],
                TriangleIndex::Second => array[1],
                TriangleIndex::Third => array[2],
            }
        }
        fn lookup_ref<'a, T>(self, array: &'a [T; 3]) -> &'a T {
            match self {
                TriangleIndex::First => &array[0],
                TriangleIndex::Second => &array[1],
                TriangleIndex::Third => &array[2],
            }
        }

        fn lookup_mut<'a, T>(self, array: &'a mut [T; 3]) -> &'a mut T {
            match self {
                TriangleIndex::First => &mut array[0],
                TriangleIndex::Second => &mut array[1],
                TriangleIndex::Third => &mut array[2],
            }
        }

        fn build<'a, T, F>(func: F) -> [T; 3]
        where
            F: Fn(TriangleIndex) -> T,
        {
            [
                func(TriangleIndex::First),
                func(TriangleIndex::Second),
                func(TriangleIndex::Third),
            ]
        }
    }

    impl Voronoi {
        pub fn size(&self) -> usize {
            self.points.len()
        }
        pub fn from_box(bounds: [Bounds; 2]) -> Voronoi {
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
            let xs = [bounds[0].min, bounds[0].max, bounds[0].max, bounds[0].min];
            let ys = [bounds[1].min, bounds[1].min, bounds[1].max, bounds[1].max];
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
        pub fn contains(&self, p: Point) -> Option<usize> {
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
                        continue;
                    }
                }
            }
        }
        fn triangle_contains(&self, i: usize, p: Point) -> bool {
            TriangleIndex::all().all(|n| self.edge_contains(i, n, p))
        }
        fn edge_contains(&self, i: usize, n: TriangleIndex, p: Point) -> bool {
            let tri = &self.triangles[i];
            let left = n.lookup(tri.vertices);
            let right = n.next().lookup(tri.vertices);
            self.points[left].ccw(self.points[right], p)
        }
        fn cos(&self, i: usize, n: TriangleIndex) -> f64 {
            let triangle = &self.triangles[i];
            let top = self.points[n.lookup(triangle.vertices)];
            let left = self.points[n.next().lookup(triangle.vertices)];
            let right = self.points[n.prev().lookup(triangle.vertices)];
            let scalar =
                (right.x - top.x) * (left.x - top.x) + (right.y - top.y) * (left.y - top.y);
            scalar / (top.dist(left) * top.dist(right))
        }
        fn split_triangle(&mut self, p: Point) -> [(usize, TriangleIndex); 3] {
            if let Some(i) = self.contains(p) {
                let triangle = &self.triangles[i];
                let verts = triangle.vertices;
                let j = self.size();
                self.points.push(p);
                let new_indices = [j, j + 1, j + 2];
                if let TriangleStatus::Alive { neighbors } = triangle.alive {
                    let mut new_triangles = TriangleIndex::build(|n| Triangle {
                        vertices: [j, n.lookup(verts), n.next().lookup(verts)],
                        alive: TriangleStatus::Alive {
                            neighbors: [
                                n.lookup(neighbors),
                                (n.next().lookup(new_indices), TriangleIndex::Third),
                                (n.prev().lookup(new_indices), TriangleIndex::Second),
                            ],
                        },
                    })
                    .to_vec();
                    self.triangles.append(&mut new_triangles);
                    for n in TriangleIndex::all() {
                        let neighbor_triangle = &self.triangles[n.lookup(neighbors).0];
                        if let TriangleStatus::Alive { mut neighbors } = neighbor_triangle.alive {
                            for n in TriangleIndex::all() {
                                *n.lookup_mut(&mut neighbors) =
                                    (n.lookup(new_indices), TriangleIndex::First);
                            }
                        } else {
                            panic!("Ran into dead neighbor")
                        }
                    }
                    self.triangles[i].alive = TriangleStatus::Dead {
                        descendants: new_indices.to_vec(),
                    };
                    [
                        (j, TriangleIndex::First),
                        (j, TriangleIndex::Second),
                        (j, TriangleIndex::First),
                    ]
                } else {
                    panic!("Splitting dead triangle")
                }
            } else {
                panic!("Splitting at exterior point")
            }
        }
        fn flip_triangle(
            &mut self,
            i: usize,
            n: TriangleIndex,
        ) -> Option<[(usize, TriangleIndex); 2]> {
            let triangle = &self.triangles[i];
            if let TriangleStatus::Alive { neighbors } = triangle.alive {
                let (i_n, n_n) = n.lookup(neighbors);
                if i_n == i || self.cos(i, n) >= -self.cos(i_n, n_n) {
                    return None;
                }
                let j = self.points.len();
                let top = i;
                let bottom = i_n;
                let left = n.next().lookup(triangle.vertices);
                let right = n.prev().lookup(triangle.vertices);
                if let TriangleStatus::Alive {
                    neighbors: neighbors_n,
                } = self.triangles[i_n].alive
                {
                    let new_indices = [j, j + 1];
                    let mut new_triangles = vec![
                        Triangle {
                            vertices: [top, left, bottom],
                            alive: TriangleStatus::Alive {
                                neighbors: [
                                    n_n.next().lookup(neighbors_n),
                                    (j + 1, TriangleIndex::Third),
                                    n.prev().lookup(neighbors),
                                ],
                            },
                        },
                        Triangle {
                            vertices: [top, bottom, right],
                            alive: TriangleStatus::Alive {
                                neighbors: [
                                    n_n.prev().lookup(neighbors_n),
                                    n.next().lookup(neighbors),
                                    (j, TriangleIndex::Second),
                                ],
                            },
                        },
                    ];
                    self.triangles.append(&mut new_triangles);
                    self.triangles[i].alive = TriangleStatus::Dead {
                        descendants: new_indices.to_vec(),
                    };
                    self.triangles[i_n].alive = TriangleStatus::Dead {
                        descendants: new_indices.to_vec(),
                    };
                    if let TriangleStatus::Alive { ref mut neighbors } = self.triangles[i].alive {
                        *n.next().lookup_mut(neighbors) = (j + 1, TriangleIndex::Second);
                        *n.prev().lookup_mut(neighbors) = (j, TriangleIndex::Third);
                    } else {
                        unreachable!()
                    };
                    if let TriangleStatus::Alive { ref mut neighbors } = self.triangles[i_n].alive {
                        *n.next().lookup_mut(neighbors) = (j, TriangleIndex::First);
                        *n.prev().lookup_mut(neighbors) = (j + 1, TriangleIndex::First);
                    } else {
                        unreachable!()
                    };
                    Some([(j, TriangleIndex::Second), (j + 1, TriangleIndex::Third)])
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        }
        pub fn insert(&mut self, p: Point) {
            let mut queue: std::collections::VecDeque<(usize, TriangleIndex)> =
                std::array::IntoIter::new(self.split_triangle(p)).collect();
            while let Some((i, n)) = queue.pop_front() {
                for &(i_new, n_new) in self.flip_triangle(i, n).iter().flatten() {
                    queue.push_back((i_new, n_new))
                }
            }
        }
        pub fn nearest_neighbour(&self, p: Point) -> (usize, f64) {
            if let Some(i_tri) = self.contains(p) {
                let triangle = &self.triangles[i_tri];
                let (d_min, i) = triangle
                    .vertices
                    .iter()
                    .map(|&i| (self.points[i].dist(p), i))
                    .fold((f64::INFINITY, 0), |(d1, i1), (d2, i2)| {
                        if d1 <= d2 {
                            (d1, i1)
                        } else {
                            (d2, i2)
                        }
                    });
                (i, d_min)
            } else {
                panic!("{:?} was outside bounds", p)
            }
        }
    }

    impl Point {
        pub fn dist(self, other: Point) -> f64 {
            ((other.x - self.x).powi(2) + (other.y - self.y).powi(2)).sqrt()
        }
        fn ccw(self, p1: Point, p2: Point) -> bool {
            return (p2.x - self.x) * (p1.y - self.y) <= (p2.y - self.y) * (p1.x - self.x);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn test_ccw_orientation() {
            let origin = Point { x: 0.0, y: 0.0 };
            let px = Point { x: 1.0, y: 0.0 };
            let py = Point { x: 0.0, y: 1.0 };
            assert_eq!(origin.ccw(px, py), true)
        }

        #[test]
        fn test_dist() {
            let origin = Point { x: 0.0, y: 0.0 };
            let px = Point { x: 1.0, y: 0.0 };
            let py = Point { x: 0.0, y: 1.0 };
            assert_eq!(origin.dist(px), 1.0);
            assert_eq!(px.dist(py), (2f64).sqrt());
        }

        #[test]
        fn test_contains() {
            let p = Point { x: 0.3, y: 0.4 };
            let q = Point { x: 0.9, y: 0.2 };
            let r = Point { x: 2.0, y: 0.5 };
            let bounds = Bounds { min: 0.0, max: 1.0 };
            let voronoi = Voronoi::from_box([bounds, bounds]);
            assert_eq!(voronoi.contains(p), Some(1));
            assert_eq!(voronoi.contains(q), Some(0));
            assert_eq!(voronoi.contains(r), None);
        }
        #[test]
        fn test_nearest_neighbour_box() {
            let p = Point { x: 0.3, y: 0.4 };
            let bounds = Bounds { min: 0.0, max: 0.0 };
            let voronoi = Voronoi::from_box([bounds, bounds]);
            assert!((0.5 - voronoi.nearest_neighbour(p).1).abs() < f64::EPSILON);
        }
    }
}
