use burn::{
    backend::{Autodiff, Metal}, nn::EmbeddingConfig, tensor::{Device, Int, Tensor}
};

fn main() {
    type MyBackend = Metal<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = Device::<MyBackend>::default();

    let input_values = [1, 3, 4, 5, 9, 3, 2, 12, 24, 2, 8, 7, 10, 11, 4, 5];
    let tensor = Tensor::<MyBackend, 1, Int>::from_data(input_values, &device);
    let embedding = EmbeddingConfig::new()
    println!("Hello, world!");
}
