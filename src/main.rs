use tch::{Tensor, Device, Kind};
use tch::{set_num_threads, set_num_interop_threads};
use rayon::prelude::*;
use std::time::Instant;
use nalgebra::dmatrix;
use nalgebra::Complex;

fn main() {
    println!("Hello, here is a template for scientific computing in Rust with tch-rs");
    // benchmark_parallelism(200, 2000, 20);
    let a = dmatrix![Complex::new(1.0, 1.0), Complex::new(2.0, 1.0), Complex::new(3.0, 1.0);
    Complex::new(4.0, 1.0), Complex::new(5.0, 1.0), Complex::new(6.0, 1.0);
    Complex::new(7.0, 1.0), Complex::new(8.0, 1.0), Complex::new(9.0, 1.0)];
    let qr = a.clone().qr();
    println!("{:?}", qr.q());
    println!("{:?}", qr.r());
    let svd = a.svd(true, true);
    println!("{:?}", svd.u);
    println!("{:?}", svd.v_t);
}

fn get_randn_tensors(tensor_size: usize, num_tensors: usize) -> Vec<Tensor> {
    let tensor_size = tensor_size as i64;
    (0..num_tensors).map(|_| Tensor::randn([tensor_size, tensor_size], (Kind::Float, Device::Cpu))).collect()
}

fn benchmark_parallelism(tensor_size: usize, num_tensors: usize, repeat: usize) {
    let core_num = num_cpus::get();
    set_num_threads(core_num as i32);
    set_num_interop_threads(core_num as i32);
    println!("core_num: {core_num}");
    rayon::ThreadPoolBuilder::new().num_threads(core_num).build_global().unwrap();
    // warmup
    let results: f64 = get_randn_tensors(tensor_size, core_num).into_par_iter().map(|t| t.sum(Kind::Float).double_value(&[])).sum();
    assert!(results != 0.0);

    let avg_time_parallel = (0..repeat).map(|_| {
        let tensors = get_randn_tensors(tensor_size, num_tensors);
        let now = Instant::now();
        let sum: f64 = tensors.into_par_iter().map(|t| t.sum(Kind::Float).double_value(&[])).sum();
        let duration = now.elapsed();
        assert!(sum != 0.0);
        duration.as_secs_f64()
    }).sum::<f64>() / repeat as f64;

    let avg_time_serial = (0..repeat).map(|_| {
        let tensors = get_randn_tensors(tensor_size, num_tensors);
        let now = Instant::now();
        let sum: f64 = tensors.into_iter().map(|t| t.sum(Kind::Float).double_value(&[])).sum();
        let duration = now.elapsed();
        assert!(sum != 0.0);
        duration.as_secs_f64()
    }).sum::<f64>() / repeat as f64;

    println!("avg_time_parallel: {avg_time_parallel}s, avg_time_serial: {avg_time_serial}s, speedup: {}x", avg_time_serial / avg_time_parallel);
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
