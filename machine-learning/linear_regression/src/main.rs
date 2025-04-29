use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    optim::AdamConfig,
};
use linear_regression::{models::taxifare_model::ModelConfig, training::TrainingConfig};
/*
fn custom_init() -> burn::backend::wgpu::WgpuSetup {
    let device = Default::default();
    init_setup::<burn::backend::wgpu::Metal>(&device, Default::default())
}
*/
fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device: WgpuDevice = Default::default();
    let artifact_dir = "../config";
    linear_regression::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(
            ModelConfig::new(vec![(7, 4), (24, 12), (2, 1)], 6, &[100, 50], 0.4),
            AdamConfig::new(),
        ),
        device.clone(),
    );
}
