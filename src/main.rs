use burn::backend::NdArray;
use burn::prelude::{Backend, Tensor as BurnTensor};
use burn::tensor::Distribution;
use clap::Parser;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;
use tch::{Device, Kind, Tensor};
use tch::{set_num_interop_threads, set_num_threads};

#[derive(Parser)]
#[command(name = "benchmark")]
#[command(about = "Benchmark scientific computing frameworks in Rust")]
struct Args {
    /// Framework to benchmark
    #[arg(value_enum)]
    framework: Framework,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Framework {
    Tch,
    Burn,
    Nalgebra,
    Candle,
    All,
}

fn main() {
    let args = Args::parse();
    let core_num = num_cpus::get();
    println!("Number of CPU cores: {core_num}");

    rayon::ThreadPoolBuilder::new()
        .num_threads(core_num)
        .build_global()
        .unwrap();

    let tensor_size = 200;
    let num_tensors = 2000;
    let repeat = 20;

    match args.framework {
        Framework::Tch => benchmark_tch(tensor_size, num_tensors, repeat, core_num),
        Framework::Burn => benchmark_burn(tensor_size, num_tensors, repeat, core_num),
        Framework::Nalgebra => benchmark_nalgebra(tensor_size, num_tensors, repeat, core_num),
        Framework::Candle => benchmark_candle(tensor_size, num_tensors, repeat, core_num),
        Framework::All => {
            benchmark_burn(tensor_size, num_tensors, repeat, core_num);
            benchmark_candle(tensor_size, num_tensors, repeat, core_num);
            benchmark_nalgebra(tensor_size, num_tensors, repeat, core_num);
            benchmark_tch(tensor_size, num_tensors, repeat, core_num);
        }
    }
}

fn benchmark_burn(tensor_size: usize, num_tensors: usize, repeat: usize, core_num: usize) {
    let burn_tensor_reduce_fn = |t: BurnTensor<NdArray, 2>| t.sum().into_scalar();
    benchmark_with_into_iter(
        tensor_size,
        num_tensors,
        repeat,
        core_num,
        get_randn_tensors_burn,
        burn_tensor_reduce_fn,
        "Inter-tensor Parallelism Speedup (Burn):",
    );
}

fn benchmark_candle(tensor_size: usize, num_tensors: usize, repeat: usize, core_num: usize) {
    let candle_tensor_reduce_fn =
        |t: &candle_core::Tensor| t.sum([0, 1]).unwrap().to_scalar::<f64>().unwrap();
    benchmark_with_iter(
        tensor_size,
        num_tensors,
        repeat,
        core_num,
        get_randn_tensors_candle,
        candle_tensor_reduce_fn,
        "Inter-tensor Parallelism Speedup (Candle):",
    );
}

fn benchmark_nalgebra(tensor_size: usize, num_tensors: usize, repeat: usize, core_num: usize) {
    let nalgebra_matrix_reduce_fn = |t: &DMatrix<f32>| t.sum();
    benchmark_with_iter(
        tensor_size,
        num_tensors,
        repeat,
        core_num,
        get_randn_tensors_nalgebra,
        nalgebra_matrix_reduce_fn,
        "Inter-tensor Parallelism Speedup (Nalgebra):",
    );
}

fn benchmark_tch(tensor_size: usize, num_tensors: usize, repeat: usize, core_num: usize) {
    let tch_tensor_reduce_fn = |t: Tensor| t.sum(Kind::Float).double_value(&[]);
    set_num_interop_threads(core_num as i32);
    set_num_threads(core_num as i32);
    benchmark_with_into_iter(
        tensor_size,
        num_tensors,
        repeat,
        core_num,
        get_randn_tensors_tch,
        tch_tensor_reduce_fn,
        "Inter-tensor Parallelism Speedup (Tch):",
    );
}

fn benchmark_with_into_iter<T: Send, F: Send + std::iter::Sum>(
    tensor_size: usize,
    num_tensors: usize,
    repeat: usize,
    core_num: usize,
    get_random_tensors_fn: impl Fn(usize, usize) -> Vec<T>,
    tensor_reduce_fn: impl Fn(T) -> F + Sync + Send + Copy,
    title: &str,
) {
    // warmup
    let _results: F = get_random_tensors_fn(tensor_size, core_num)
        .into_par_iter()
        .map(tensor_reduce_fn)
        .sum();
    black_box(_results);
    // use into_par_iter will reduce speedup from 11x to 8x
    let avg_time_parallel = (0..repeat)
        .map(|_| {
            let tensors = get_random_tensors_fn(tensor_size, num_tensors);
            let now = Instant::now();
            let _sum: F = tensors.into_par_iter().map(tensor_reduce_fn).sum();
            let duration = now.elapsed();
            black_box(_sum);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    let avg_time_serial = (0..repeat)
        .map(|_| {
            let tensors = get_random_tensors_fn(tensor_size, num_tensors);
            let now = Instant::now();
            let _sum: F = tensors.into_iter().map(tensor_reduce_fn).sum();
            let duration = now.elapsed();
            black_box(_sum);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    println!(
        "{title}\n avg_time_parallel: {avg_time_parallel}s\n avg_time_serial: {avg_time_serial}s\n speedup: {}x",
        avg_time_serial / avg_time_parallel
    );
}

fn benchmark_with_iter<T: Send + Sync, F: Send + std::iter::Sum>(
    tensor_size: usize,
    num_tensors: usize,
    repeat: usize,
    core_num: usize,
    get_random_tensors_fn: impl Fn(usize, usize) -> Vec<T>,
    tensor_reduce_fn: impl Fn(&T) -> F + Sync + Send + Copy,
    title: &str,
) {
    // warmup
    let _results: F = get_random_tensors_fn(tensor_size, core_num)
        .par_iter()
        .map(tensor_reduce_fn)
        .sum();
    black_box(_results);
    // use into_par_iter will reduce speedup from 11x to 8x
    let avg_time_parallel = (0..repeat)
        .map(|_| {
            let tensors = get_random_tensors_fn(tensor_size, num_tensors);
            let now = Instant::now();
            let _sum: F = tensors.par_iter().map(tensor_reduce_fn).sum();
            let duration = now.elapsed();
            black_box(_sum);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    let avg_time_serial = (0..repeat)
        .map(|_| {
            let tensors = get_random_tensors_fn(tensor_size, num_tensors);
            let now = Instant::now();
            let _sum: F = tensors.iter().map(tensor_reduce_fn).sum();
            let duration = now.elapsed();
            black_box(_sum);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    println!(
        "{title}\n avg_time_parallel: {avg_time_parallel}s\n avg_time_serial: {avg_time_serial}s\n speedup: {}x",
        avg_time_serial / avg_time_parallel
    );
}

fn get_randn_tensors_burn(tensor_size: usize, num_tensors: usize) -> Vec<BurnTensor<NdArray, 2>> {
    let distr = Distribution::Normal(1.0, 0.0);
    let device = <NdArray<f32, i64, i8> as Backend>::Device::default();
    (0..num_tensors)
        .map(|_| BurnTensor::<NdArray, 2>::random([tensor_size, tensor_size], distr, &device))
        .collect()
}

fn get_randn_tensors_tch(tensor_size: usize, num_tensors: usize) -> Vec<Tensor> {
    let tensor_size = tensor_size as i64;
    (0..num_tensors)
        .map(|_| Tensor::randn([tensor_size, tensor_size], (Kind::Float, Device::Cpu)))
        .collect()
}

fn get_randn_tensors_nalgebra(tensor_size: usize, num_tensors: usize) -> Vec<DMatrix<f32>> {
    let standard_distribution = rand::distributions::Standard;
    let mut rng = rand::thread_rng();
    (0..num_tensors)
        .map(|_| {
            DMatrix::from_distribution(tensor_size, tensor_size, &standard_distribution, &mut rng)
        })
        .collect()
}

fn get_randn_tensors_candle(tensor_size: usize, num_tensors: usize) -> Vec<candle_core::Tensor> {
    let device = candle_core::Device::Cpu;
    (0..num_tensors)
        .map(|_| {
            candle_core::Tensor::randn(0.0, 1.0, &[tensor_size, tensor_size], &device).unwrap()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use tch::nn::{Adam, OptimizerConfig, VarStore};
    use tch::{Device, Kind, Tensor};

    #[test]
    fn it_works() {
        let a = Tensor::ones([2, 2], (Kind::Float, Device::Cpu));
        let b = a * 2. * 2.;
        let result = b.sum(Kind::Float).double_value(&[]);
        assert_eq!(result, 2. * 2. * (2. * 2.));
    }

    #[test]
    fn test_optimizer() {
        let vs = VarStore::new(Device::Cpu);
        let root = vs.root();
        let mut adam = Adam::default().build(&vs, 1e-1).unwrap();
        let v = root.randn("v", &[2, 2, 2, 2], 0.0, 1.0);
        let v_copy = v.copy();
        v_copy.print();
        let loss = v.pow_tensor_scalar(2).sum(Kind::Float);
        loss.backward();
        let gradient = v.grad();
        assert!(gradient.allclose(&(2 * v.copy()), 1e-5, 1e-5, false));
        adam.step();
        adam.zero_grad();
        v.print();
        assert!(!v.allclose(&v_copy, 1e-5, 1e-5, false));
    }
}
