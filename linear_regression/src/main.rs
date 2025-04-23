use burn::backend::{Wgpu, wgpu::WgpuDevice};
use linear_regression::data::create_input_dataset;

fn main() {
    let merged_df = create_input_dataset("NYCTaxiFares.csv").unwrap();
    println!("Columns: {:?}", merged_df.get_column_names());
    println!("{:?}", merged_df);

    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();
}
