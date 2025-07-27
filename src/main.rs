use burn::backend::NdArray;
use burn::prelude::{Backend, Tensor as BurnTensor};
use burn::tensor::Distribution;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::time::Instant;
use tch::{set_num_interop_threads, set_num_threads};
use tch::{Device, Kind, Tensor};

fn main() {
    println!("Hello, here is a template for scientific computing in Rust with tch-rs");
    let tensor_size = 200;
    let num_tensors = 2000;
    let repeat = 20;
    // benchmark_parallelism_tch(tensor_size, num_tensors, repeat);
    // benchmark_parallelism_nalgebra(tensor_size, num_tensors, repeat);
    // benchmark_parallelism_burn(tensor_size, num_tensors, repeat);
    benchmark_parallelism_candle(tensor_size, num_tensors, repeat);
    // let a = dmatrix![Complex::new(1.0, 1.0), Complex::new(2.0, 1.0), Complex::new(3.0, 1.0);
    // Complex::new(4.0, 1.0), Complex::new(5.0, 1.0), Complex::new(6.0, 1.0);
    // Complex::new(7.0, 1.0), Complex::new(8.0, 1.0), Complex::new(9.0, 1.0)];
    // let qr = a.clone().qr();
    // println!("{:?}", qr.q());
    // println!("{:?}", qr.r());
    // let svd = a.svd(true, true);
    // println!("{:?}", svd.u);
    // println!("{:?}", svd.v_t);
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

fn benchmark_parallelism_candle(tensor_size: usize, num_tensors: usize, repeat: usize) {
    let core_num = num_cpus::get();
    println!("core_num: {core_num}");
    rayon::ThreadPoolBuilder::new()
        .num_threads(core_num)
        .build_global()
        .unwrap();
    // warmup
    let results: f64 = get_randn_tensors_candle(tensor_size, core_num)
        .par_iter()
        .map(|t| t.sum([0, 1]).unwrap().to_scalar::<f64>().unwrap())
        .sum();
    assert!(results != 0.0);
    // use into_par_iter will reduce speedup from 11x to 8x
    let avg_time_parallel = (0..repeat)
        .map(|_| {
            let tensors = get_randn_tensors_candle(tensor_size, num_tensors);
            let now = Instant::now();
            let sum: f64 = tensors.par_iter().map(|t| t.sum([0, 1]).unwrap().to_scalar::<f64>().unwrap()).sum();
            let duration = now.elapsed();
            assert!(sum != 0.0);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    let avg_time_serial = (0..repeat)
        .map(|_| {
            let tensors = get_randn_tensors_candle(tensor_size, num_tensors);
            let now = Instant::now();
            let sum: f64 = tensors.iter().map(|t| t.sum([0, 1]).unwrap().to_scalar::<f64>().unwrap()).sum();
            let duration = now.elapsed();
            assert!(sum != 0.0);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    println!(
        "avg_time_parallel: {avg_time_parallel}s, avg_time_serial: {avg_time_serial}s, speedup: {}x",
        avg_time_serial / avg_time_parallel
    );
}

fn benchmark_parallelism_burn(tensor_size: usize, num_tensors: usize, repeat: usize) {
    let core_num = num_cpus::get();
    println!("core_num: {core_num}");
    rayon::ThreadPoolBuilder::new()
        .num_threads(core_num)
        .build_global()
        .unwrap();
    // warmup
    let results: f32 = get_randn_tensors_burn(tensor_size, core_num)
        .into_par_iter()
        .map(|t| t.sum().into_scalar())
        .sum();
    assert!(results != 0.0);
    // use into_par_iter will reduce speedup from 11x to 8x
    let avg_time_parallel = (0..repeat)
        .map(|_| {
            let tensors = get_randn_tensors_burn(tensor_size, num_tensors);
            let now = Instant::now();
            let sum: f32 = tensors.into_par_iter().map(|t| t.sum().into_scalar()).sum();
            let duration = now.elapsed();
            assert!(sum != 0.0);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    let avg_time_serial = (0..repeat)
        .map(|_| {
            let tensors = get_randn_tensors_burn(tensor_size, num_tensors);
            let now = Instant::now();
            let sum: f32 = tensors.into_iter().map(|t| t.sum().into_scalar()).sum();
            let duration = now.elapsed();
            assert!(sum != 0.0);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    println!(
        "avg_time_parallel: {avg_time_parallel}s, avg_time_serial: {avg_time_serial}s, speedup: {}x",
        avg_time_serial / avg_time_parallel
    );
}

fn benchmark_parallelism_nalgebra(tensor_size: usize, num_tensors: usize, repeat: usize) {
    let core_num = num_cpus::get();
    println!("core_num: {core_num}");
    rayon::ThreadPoolBuilder::new()
        .num_threads(core_num)
        .build_global()
        .unwrap();
    // warmup
    let results: f32 = get_randn_tensors_nalgebra(tensor_size, core_num)
        .par_iter()
        .map(|t| t.sum())
        .sum();
    assert!(results != 0.0);
    // use into_par_iter will reduce speedup from 11x to 8x
    let avg_time_parallel = (0..repeat)
        .map(|_| {
            let tensors = get_randn_tensors_nalgebra(tensor_size, num_tensors);
            let now = Instant::now();
            let sum: f32 = tensors.par_iter().map(|t| t.sum()).sum();
            let duration = now.elapsed();
            assert!(sum != 0.0);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    let avg_time_serial = (0..repeat)
        .map(|_| {
            let tensors = get_randn_tensors_nalgebra(tensor_size, num_tensors);
            let now = Instant::now();
            let sum: f32 = tensors.iter().map(|t| t.sum()).sum();
            let duration = now.elapsed();
            assert!(sum != 0.0);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    println!(
        "avg_time_parallel: {avg_time_parallel}s, avg_time_serial: {avg_time_serial}s, speedup: {}x",
        avg_time_serial / avg_time_parallel
    );
}

fn benchmark_parallelism_tch(tensor_size: usize, num_tensors: usize, repeat: usize) {
    let core_num = num_cpus::get();
    set_num_threads(core_num as i32);
    set_num_interop_threads(core_num as i32);
    println!("core_num: {core_num}");
    rayon::ThreadPoolBuilder::new()
        .num_threads(core_num)
        .build_global()
        .unwrap();
    // warmup
    let results: f64 = get_randn_tensors_tch(tensor_size, core_num)
        .into_par_iter()
        .map(|t| t.sum(Kind::Float).double_value(&[]))
        .sum();
    assert!(results != 0.0);

    let avg_time_parallel = (0..repeat)
        .map(|_| {
            let tensors = get_randn_tensors_tch(tensor_size, num_tensors);
            let now = Instant::now();
            let sum: f64 = tensors
                .into_par_iter()
                .map(|t| t.sum(Kind::Float).double_value(&[]))
                .sum();
            let duration = now.elapsed();
            assert!(sum != 0.0);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    let avg_time_serial = (0..repeat)
        .map(|_| {
            let tensors = get_randn_tensors_tch(tensor_size, num_tensors);
            let now = Instant::now();
            let sum: f64 = tensors
                .into_iter()
                .map(|t| t.sum(Kind::Float).double_value(&[]))
                .sum();
            let duration = now.elapsed();
            assert!(sum != 0.0);
            duration.as_secs_f64()
        })
        .sum::<f64>()
        / repeat as f64;

    println!(
        "avg_time_parallel: {avg_time_parallel}s, avg_time_serial: {avg_time_serial}s, speedup: {}x",
        avg_time_serial / avg_time_parallel
    );
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
        adam.step();
        adam.zero_grad();
        v.print();
        assert!(!v.allclose(&v_copy, 1e-5, 1e-5, false));
    }
}
