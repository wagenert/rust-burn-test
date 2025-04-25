use burn::{
    backend::{Wgpu, wgpu::WgpuDevice},
    tensor::{Shape, Tensor},
};
use linear_regression::{data::create_input_dataset, models::ModelConfig};
use polars::prelude::*;

/*
fn custom_init() -> burn::backend::wgpu::WgpuSetup {
    let device = Default::default();
    init_setup::<burn::backend::wgpu::Metal>(&device, Default::default())
}
*/

fn main() {
    let merged_df = create_input_dataset("NYCTaxiFares.csv").unwrap();
    let embedding_df = merged_df
        .clone()
        .lazy()
        .select([col("pickup_hour"), col("pickup_weekday")])
        .collect()
        .unwrap();
    let continuous_df = merged_df
        .clone()
        .lazy()
        .select([
            col("pickup_latitude"),
            col("pickup_longitude"),
            col("dropoff_latitude"),
            col("dropoff_longitude"),
            col("passenger_count"),
            col("distance"),
        ])
        .collect()
        .unwrap();
    println!("Columns: {:?}", merged_df.get_column_names());
    println!("{:?}", merged_df.head(Some(5)));
    let dims = continuous_df.shape();
    let dims_as_slice = [dims.0, dims.1];
    let cols = continuous_df
        .get_columns()
        .iter()
        .flat_map(|col| {
            col.clone()
                .f64()
                .unwrap()
                .to_vec_null_aware()
                .left()
                .unwrap()
        })
        .collect::<Vec<f64>>();
    //.collect::<Vec<Vec<f64>>>();
    //let wgpu_setup = custom_init();
    type MyBackend = Wgpu<f32, i32>;
    let device: WgpuDevice = Default::default();

    let continuous_tensor: Tensor<MyBackend, 2> =
        Tensor::<MyBackend, 1>::from(cols.as_slice()).reshape::<2, _>(Shape::new(dims_as_slice));
    let model = ModelConfig::new(vec![(0, 1), (1, 1), (2, 1)], 3, &[64, 32], 0.4)
        .init::<MyBackend>(&device.clone());

    println!("Model: {:?}", model);
}
