pub mod voronoi {
    use std::iter;
    use svg::node::element::path::Data;
    use svg::node::element::Path;
    use svg::Document;
    #[derive(Copy, Clone, Debug)]
    pub struct Point {
        x: f64,
        y: f64,
    }

    #[derive(Clone, Debug, Copy)]
    enum TriangleStatus {
        Alive {
            neighbors: [Option<(usize, TriangleIndex)>; 3],
        },
        Dead {
            start: usize,
            end: usize,
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
    pub struct Delauney {
        points: Vec<Point>,
        triangles: Vec<Triangle>,
        bounds: [Bounds; 2],
    }
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
        fn _lookup_ref<'a, T>(self, array: &'a [T; 3]) -> &'a T {
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

    impl Delauney {
        pub fn size(&self) -> usize {
            self.points.len()
        }
        pub fn from_box(bounds: [Bounds; 2]) -> Delauney {
            let bottom = Triangle {
                vertices: [0, 1, 2],
                alive: TriangleStatus::Alive {
                    neighbors: [None, Some((1, TriangleIndex::Third)), None],
                },
            };
            let top = Triangle {
                vertices: [0, 2, 3],
                alive: TriangleStatus::Alive {
                    neighbors: [None, None, Some((0, TriangleIndex::Second))],
                },
            };
            let xs = [bounds[0].min, bounds[0].max, bounds[0].max, bounds[0].min];
            let ys = [bounds[1].min, bounds[1].min, bounds[1].max, bounds[1].max];
            let points = xs
                .iter()
                .zip(ys.iter())
                .map(|(&x, &y)| Point { x, y })
                .collect();
            Delauney {
                points,
                triangles: vec![bottom, top],
                bounds,
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
                match self.triangles[i].alive {
                    TriangleStatus::Alive { .. } => return Some(i),
                    TriangleStatus::Dead { start, end } => {
                        for j in start..end {
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

        fn split_triangle(&mut self, p: Point) -> Option<[(usize, TriangleIndex); 3]> {
            if let Some(i) = self.contains(p) {
                let triangle = &self.triangles[i];
                let verts = triangle.vertices;
                let j = self.triangles.len();
                let k = self.size();
                self.points.push(p);
                let new_indices = [j, j + 1, j + 2];
                if let TriangleStatus::Alive {
                    neighbors: old_neighbors,
                } = triangle.alive
                {
                    let mut new_triangles = TriangleIndex::build(|n| Triangle {
                        vertices: [k, n.next().lookup(verts), n.prev().lookup(verts)],
                        alive: TriangleStatus::Alive {
                            neighbors: [
                                None,
                                Some((n.next().lookup(new_indices), TriangleIndex::Third)),
                                Some((n.prev().lookup(new_indices), TriangleIndex::Second)),
                            ],
                        },
                    })
                    .to_vec();
                    self.triangles.append(&mut new_triangles);
                    for n in TriangleIndex::all() {
                        self.link_half_edges(
                            (n.lookup(new_indices), TriangleIndex::First),
                            n.lookup(old_neighbors),
                        );
                    }
                    self.triangles[i].alive = TriangleStatus::Dead {
                        start: j,
                        end: j + 3,
                    };
                    Some([
                        (j, TriangleIndex::First),
                        (j + 1, TriangleIndex::First),
                        (j + 2, TriangleIndex::First),
                    ])
                } else {
                    panic!("Splitting dead triangle")
                }
            } else {
                None
            }
        }

        fn link_half_edges(
            &mut self,
            start: (usize, TriangleIndex),
            end_opt: Option<(usize, TriangleIndex)>,
        ) -> () {
            if let TriangleStatus::Alive { ref mut neighbors } = self.triangles[start.0].alive {
                *(start.1).lookup_mut(neighbors) = end_opt;
            } else {
                unreachable!()
            };
            if let Some(end) = end_opt {
                if let TriangleStatus::Alive { ref mut neighbors } = self.triangles[end.0].alive {
                    *(end.1).lookup_mut(neighbors) = Some(start);
                } else {
                    unreachable!()
                }
            }
        }

        fn flip_triangle(
            &mut self,
            i: usize,
            n: TriangleIndex,
        ) -> Option<[(usize, TriangleIndex); 2]> {
            let triangle = &self.triangles[i];
            if let TriangleStatus::Alive { neighbors } = triangle.alive {
                if let Some((i_n, n_n)) = n.lookup(neighbors) {
                    if self.cos(i, n) >= -self.cos(i_n, n_n) {
                        return None;
                    }
                    let j = self.triangles.len();
                    let top = n.lookup(triangle.vertices);
                    let bottom = n_n.lookup(self.triangles[i_n].vertices);
                    let left = n.next().lookup(triangle.vertices);
                    let right = n.prev().lookup(triangle.vertices);
                    if let TriangleStatus::Alive {
                        neighbors: neighbors_n,
                    } = self.triangles[i_n].alive
                    {
                        debug_assert_eq!(n_n.lookup(neighbors_n), Some((i, n)));
                        let mut new_triangles = vec![
                            Triangle {
                                vertices: [top, left, bottom],
                                alive: TriangleStatus::Alive {
                                    neighbors: [None, Some((j + 1, TriangleIndex::Third)), None],
                                },
                            },
                            Triangle {
                                vertices: [top, bottom, right],
                                alive: TriangleStatus::Alive {
                                    neighbors: [None, None, Some((j, TriangleIndex::Second))],
                                },
                            },
                        ];
                        self.triangles.append(&mut new_triangles);

                        self.link_half_edges(
                            (j, TriangleIndex::First),
                            n_n.next().lookup(neighbors_n),
                        );
                        self.link_half_edges((j, TriangleIndex::Third), n.prev().lookup(neighbors));
                        self.link_half_edges(
                            (j + 1, TriangleIndex::First),
                            n_n.prev().lookup(neighbors_n),
                        );
                        self.link_half_edges(
                            (j + 1, TriangleIndex::Second),
                            n.next().lookup(neighbors),
                        );

                        self.triangles[i].alive = TriangleStatus::Dead {
                            start: j,
                            end: j + 2,
                        };
                        self.triangles[i_n].alive = TriangleStatus::Dead {
                            start: j,
                            end: j + 2,
                        };
                        Some([(j, TriangleIndex::First), (j + 1, TriangleIndex::First)])
                    } else {
                        unreachable!()
                    }
                } else {
                    None
                }
            } else {
                unreachable!()
            }
        }

        pub fn insert(&mut self, p: Point) {
            let mut queue = std::collections::VecDeque::<(usize, TriangleIndex)>::new();
            for &(i_new, n_new) in self.split_triangle(p).iter().flatten() {
                queue.push_back((i_new, n_new))
            }
            while let Some((i, n)) = queue.pop_front() {
                let _x = dbg!(&self.triangles);
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

        fn generate_triangle(&self, i: usize) -> Option<Path> {
            match self.triangles[i].alive {
                TriangleStatus::Dead { .. } => None,
                TriangleStatus::Alive { .. } => {
                    let points: Vec<(f64, f64)> = self.triangles[i]
                        .vertices
                        .iter()
                        .map(|k| self.points[*k].into())
                        .collect();
                    let data = Data::new()
                        .move_to(points[0])
                        .line_to(points[1])
                        .line_to(points[2])
                        .close();
                    let path = Path::new()
                        .set("fill", "none")
                        .set("stroke", "black")
                        .set("stroke-width", 1)
                        .set("stroke-linejoin", "round")
                        .set("d", data);
                    Some(path)
                }
            }
        }

        fn save_svg(&self, filename: String) -> () {
            let mut document = Document::new().set(
                "viewBox",
                (
                    self.bounds[0].min,
                    self.bounds[1].min,
                    self.bounds[0].max - self.bounds[0].min,
                    self.bounds[1].max - self.bounds[1].min,
                ),
            );
            for i in 0..(self.triangles.len()) {
                if let Some(p) = self.generate_triangle(i) {
                    document = document.add(p);
                }
            }
            let _x = dbg!(&filename);
            svg::save(filename, &document).unwrap();
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

    impl From<Point> for (f64, f64) {
        fn from(p: Point) -> (f64, f64) {
            (p.x, p.y)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use rand::prelude::*;
        use rand_pcg::Pcg64;

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
            let delauney = Delauney::from_box([bounds, bounds]);
            assert_eq!(delauney.contains(p), Some(1));
            assert_eq!(delauney.contains(q), Some(0));
            assert_eq!(delauney.contains(r), None);
        }
        #[test]
        fn test_nearest_neighbour_box() {
            let p = Point { x: 0.3, y: 0.4 };
            let bounds = Bounds { min: 0.0, max: 1.0 };
            let delauney = Delauney::from_box([bounds, bounds]);
            assert!((0.5 - delauney.nearest_neighbour(p).1).abs() < f64::EPSILON);
        }
        #[test]
        fn test_ccw_insert() {
            let p = Point { x: 0.3, y: 0.4 };
            let bounds = Bounds { min: 0.0, max: 0.0 };
            let mut delauney = Delauney::from_box([bounds, bounds]);
            delauney.insert(p);
            for i in 0..(delauney.triangles.len()) {
                for n in TriangleIndex::all() {
                    assert!(delauney.edge_contains(
                        i,
                        n,
                        delauney.points[n.prev().lookup(delauney.triangles[i].vertices)]
                    ))
                }
            }
        }

        #[test]
        fn test_insert_inside_outside() {
            let p = Point { x: 0.3, y: 0.4 };
            let q = Point { x: 0.9, y: 0.2 };
            let r = Point { x: 2.0, y: 0.5 };
            let bounds = Bounds { min: 0.0, max: 1.0 };
            let mut delauney = Delauney::from_box([bounds, bounds]);

            delauney.insert(p);
            assert!(delauney.contains(q).is_some());
            delauney.insert(q);
            assert_eq!(delauney.size(), 6);
            delauney.insert(r);
            assert_eq!(delauney.points.len(), 6);
        }

        #[test]
        fn test_insert_many() {
            const N: usize = 1000;
            const SEED: u64 = 333;
            let mut rng = Pcg64::seed_from_u64(SEED);

            let bounds = Bounds {
                min: -100.0,
                max: 1200.0,
            };

            let mut delauney = Delauney::from_box([bounds, bounds]);
            for i in 0..N {
                let p = Point {
                    x: rng.gen_range(0.0..1000.0),
                    y: rng.gen_range(0.0..1000.0),
                };
                if 0 == (i + 1) % 10 {
                    delauney.save_svg(format!("./test{}.svg", i))
                };
                delauney.insert(p)
            }
        }
    }
}
