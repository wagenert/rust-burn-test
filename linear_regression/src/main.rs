use burn::{
    backend::{Wgpu, wgpu::WgpuDevice},
    tensor::{Int, Shape, Tensor},
};
use linear_regression::{data::create_input_dataset, models::ModelConfig};
use polars::prelude::*;

/*
fn custom_init() -> burn::backend::wgpu::WgpuSetup {
    let device = Default::default();
    init_setup::<burn::backend::wgpu::Metal>(&device, Default::default())
}
*/

fn get_continuous_columns(df: &DataFrame) -> (Vec<f64>, [usize; 2]) {
    let cont_cols = df
        .clone()
        .lazy()
        .select([
            col("pickup_latitude"),
            col("pickup_longitude"),
            col("dropoff_latitude"),
            col("dropoff_longitude"),
            col("passenger_count").cast(DataType::Float64),
            col("distance"),
        ])
        .collect()
        .unwrap();

    let dims = cont_cols.shape();
    let dims_as_slice = [dims.0, dims.1];
    let cols = cont_cols
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
    (cols, dims_as_slice)
}

fn get_embedding_columns(df: &DataFrame) -> (Vec<Vec<i32>>, Vec<(usize, usize)>) {
    let embedding_vec = df
        .columns(["pickup_hour", "pickup_weekday", "am_or_pm"])
        .unwrap()
        .iter()
        .map(|col| {
            col.cast(&DataType::Int32)
                .unwrap()
                .i32()
                .unwrap()
                .to_vec_null_aware()
                .left()
                .unwrap()
        })
        .collect::<Vec<Vec<i32>>>();
    let embedding_df = df
        .clone()
        .lazy()
        .select([col("pickup_hour"), col("pickup_weekday"), col("am_or_pm")])
        .collect()
        .unwrap();
    let embedding_cats = embedding_df
        .get_columns()
        .iter()
        .map(|col| {
            let no_of_cats = col.unique().unwrap().len();
            (no_of_cats, usize::min(no_of_cats / 2, 50))
        })
        .collect::<Vec<(usize, usize)>>();
    (embedding_vec, embedding_cats)
}

fn get_target_vector(df: &DataFrame) -> Vec<f64> {
    df.column("fare_amount")
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap()
        .f64()
        .unwrap()
        .to_vec_null_aware()
        .left()
        .unwrap()
}
fn main() {
    let merged_df = create_input_dataset("NYCTaxiFares.csv").unwrap();
    let (embedding_vec, embedding_cats) = get_embedding_columns(&merged_df);
    println!("Embedding categories: {:?}", embedding_cats);
    let (continuous_vec, continuous_dims) = get_continuous_columns(&merged_df);
    let targets_vec = get_target_vector(&merged_df);
    type MyBackend = Wgpu<f32, i32>;
    let device: WgpuDevice = Default::default();

    let continuous_tensor: Tensor<MyBackend, 2> =
        Tensor::<MyBackend, 1>::from(continuous_vec.as_slice())
            .reshape::<2, _>(Shape::new(continuous_dims));
    let embedding_tensor: Vec<Tensor<MyBackend, 2, Int>> = embedding_vec
        .iter()
        .map(|col| Tensor::<MyBackend, 1, Int>::from(col.as_slice()).unsqueeze())
        .collect();

    let targets_tensor: Tensor<MyBackend, 1> = Tensor::<MyBackend, 1>::from(targets_vec.as_slice());
    let model = ModelConfig::new(embedding_cats, continuous_dims[1], &[64, 32], 0.4)
        .init::<MyBackend>(&device.clone());

    //let y_pred = model.forward(embedding_tensor, continuous_tensor);
    //println!("Output: {:?}", y_pred);
    println!("Model: {:?}", model);
}
