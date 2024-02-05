#![feature(portable_simd)]

use clap::Parser;
use image::{Pixel, Rgb, RgbImage};
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::{path::PathBuf, simd::prelude::*};
use std::time::Instant;

const LANES: usize = 8;
type Reg = Simd<f64, LANES>;

#[derive(Clone, Copy)]
struct SimdComplex {
    re: Reg,
    im: Reg,
}

impl SimdComplex {
    /// Returns (self * self) + c
    fn square_add(self, c: Self) -> Self {
        let re = self.re * self.re - self.im * self.im + c.re;
        let im = Reg::splat(2.0) * self.re * self.im + c.im;
        Self { re, im }
    }

    fn norm2(self) -> Reg {
        self.re * self.re + self.im * self.im
    }
}

fn mandelbrot_iterations_simd(c: SimdComplex, max_iterations: u32) -> [u32; LANES] {
    let mut z = SimdComplex {
        re: Reg::splat(0.0),
        im: Reg::splat(0.0),
    };
    type U32Reg = Simd<u32, LANES>;
    let mut iterations = U32Reg::splat(max_iterations);

    for i in 0..max_iterations {
        let mask = z.norm2().simd_ge(Reg::splat(4.0)).cast();
        let current_iteration = mask.select(U32Reg::splat(i + 1), U32Reg::splat(max_iterations));
        iterations = iterations.simd_min(current_iteration);
        z = z.square_add(c);
        if mask.all() {
            break;
        }
    }

    iterations.to_array()
}

fn mandelbrot_iterations(vals: (f64, f64), max_iterations: u32) -> u32 {
    let mut z = (0.0, 0.0);
    for i in 0..max_iterations {
        z = (z.0 * z.0 - z.1 * z.1 + vals.0, 2.0 * z.0 * z.1 + vals.1);
        if z.0 * z.0 + z.1 * z.1 >= 4.0 {
            return i + 1;
        }
    }
    max_iterations
}

pub struct Grid {
    pub width: u32,
    pub height: u32,
    pub pixels: RgbImage,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl Grid {
    pub fn new(width: u32, height: u32, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self {
            width,
            height,
            pixels: RgbImage::new(width, height),
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    /// converts an hsv color (h in [0, 360], s in [0, 1], v in [0, 1])
    /// to an rgb color (r, g, b in [0, 255]).
    pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Rgb<u8> {
        let f = |n: f32| {
            let k = (n + h / 60.0) % 6.0;
            v - v * s * 0f32.max(k.min(4.0 - k).min(1.0))
        };
        let rgb = [f(5.0) * 255.99, f(3.0) * 255.99, f(1.0) * 255.99];
        Rgb([rgb[0] as u8, rgb[1] as u8, rgb[2] as u8])
    }

    pub fn color_from_iterations(its: u32, max_iterations: u32) -> Rgb<u8> {
        if its == max_iterations {
            Rgb([0, 0, 0])
        } else {
            // hsv = [powf((i / max) * 360, 1.5) % 360, 100, (i / max) * 100]
            let r = (its as f32) / (max_iterations as f32) * 360.0;
            let h = r;
            let s = 1.0;
            let v = f32::from(its < max_iterations);
            Self::hsv_to_rgb(h, s, v)
        }
    }

    pub fn compute(&mut self, max_iterations: u32) {
        let (width, height, x_min, x_max, y_min, y_max) = (
            self.width,
            self.height,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
        );

        let pix_to_coord = |x: u32, y: u32| {
            let x = (x as f64) / width as f64;
            let y = ((height - y) as f64) / height as f64;
            (x_min + (x_max - x_min) * x, y_min + (y_max - y_min) * y)
        };

        self.pixels
            .par_chunks_exact_mut(3 * self.width as usize)
            .enumerate()
            .progress()
            .with_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] {bar:60.green} {human_pos}/{human_len} {msg} ({percent}%)",
                )
                .unwrap(),
            )
            .with_message("rows rendered")
            .for_each(|(row, data)| {
                let row = row as u32;
                let mut col = 0;
                while col + LANES as u32 <= self.width {
                    let x_off = (x_max - x_min) / width as f64;
                    let (x1, y) = pix_to_coord(col, row);
                    let offsets = Reg::from_array(std::array::from_fn(|i| x_off * (i as f64)));
                    let coords = SimdComplex {
                        re: Reg::splat(x1) + offsets,
                        im: Reg::splat(y),
                    };
                    let res = mandelbrot_iterations_simd(coords, max_iterations);
                    for (i, &its) in res.iter().enumerate() {
                        let idx = 3 * (col as usize + i);
                        let px = Rgb::from_slice_mut(&mut data[idx..idx + 3]);
                        let color = Self::color_from_iterations(its, max_iterations);
                        *px = color;
                    }
                    col += LANES as u32;
                }
                for col in col..width {
                    let (x, y) = pix_to_coord(col, row);
                    let idx = 3 * col as usize;
                    let px = Rgb::from_slice_mut(&mut data[idx..idx + 3]);
                    let its = mandelbrot_iterations((x, y), max_iterations);
                    let color = Self::color_from_iterations(its, max_iterations);
                    *px = color;
                }
            });
    }
}

#[derive(Parser)]
struct Args {
    #[clap(short, long)]
    columns: u32,
    #[clap(short, long)]
    rows: u32,
    #[clap(short, default_value = "-0.74936425")]
    x: f64,
    #[clap(short, default_value = "0.0316384815")]
    y: f64,
    #[clap(short, long, default_value = "0.0002475")]
    width: f64,
    #[clap(short, long, default_value = "0.000141897")]
    height: f64,
    #[clap(short = 'M', long, default_value = "4096")]
    max_iterations: u32,
    #[clap(short, long, default_value = "mandelbrot.png")]
    output: PathBuf,
}

fn main() {
    let Args {
        columns,
        rows,
        x,
        y,
        width,
        height,
        max_iterations,
        output,
    } = Args::parse();

    let mut grid = Grid::new(
        columns,
        rows,
        x - width / 2.0,
        x + width / 2.0,
        y - height / 2.0,
        y + height / 2.0,
    );
    let t = Instant::now();
    grid.compute(max_iterations);
    println!("Time: {:.3?}", t.elapsed());
    let img = grid.pixels;
    img.save(&output).unwrap();
}
