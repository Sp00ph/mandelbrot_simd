
## SIMD-accelerated multithreaded mandelbrot renderer

Requires a recent nightly rustc to run. Example usage:

```sh
cargo r --release -- -c 3000 -r 2000 -x="-0.5" -y 0 -w 3 -h 2 -M 100
```

If your CPU supports 512-bit vectors, changing the `LANES` constant in `src/main.rs` from 8 to 16 may improve rendering speed.