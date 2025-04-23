use burn::backend::{Wgpu, wgpu::WgpuDevice};
use linear_regression::{data::create_input_dataset, models::ModelConfig};

fn main() {
    let merged_df = create_input_dataset("NYCTaxiFares.csv").unwrap();
    println!("Columns: {:?}", merged_df.get_column_names());
    println!("{:?}", merged_df.head(Some(5)));

    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let model = ModelConfig::new(vec![(0, 1), (1, 1), (2, 1)], 3, &[64, 32], 0.4)
        .init::<MyBackend>(&device);

    println!("Model: {:?}", model);
}
