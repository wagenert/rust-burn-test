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

fn main() {
    let merged_df = create_input_dataset("NYCTaxiFares.csv").unwrap();
    let embedding_df = merged_df
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

    println!("Embedding categories: {:?}", embedding_cats);

    let embedding_shape = embedding_df.shape();
    let embedding_shape_as_slice = [embedding_shape.0, embedding_shape.1];
    let embedding = embedding_df
        .get_columns()
        .iter()
        .flat_map(|col| {
            col.clone()
                .cast(&DataType::Int32)
                .unwrap()
                .i32()
                .unwrap()
                .to_vec_null_aware()
                .left()
                .unwrap()
        })
        .collect::<Vec<i32>>();

    //println!("Columns: {:?}", merged_df.get_column_names());
    //println!("{:?}", merged_df.head(Some(5)));
    let (cols, dims) = get_continuous_columns(&merged_df);
    // let wgpu_setup = custom_init();
    let targets = merged_df
        .column("fare_amount")
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap()
        .f64()
        .unwrap()
        .to_vec_null_aware()
        .left()
        .unwrap();
    type MyBackend = Wgpu<f32, i32>;
    let device: WgpuDevice = Default::default();

    let continuous_tensor: Tensor<MyBackend, 2> =
        Tensor::<MyBackend, 1>::from(cols.as_slice()).reshape::<2, _>(Shape::new(dims));
    let embedding_tensor: Tensor<MyBackend, 2, Int> =
        Tensor::<MyBackend, 1, Int>::from(embedding.as_slice())
            .reshape::<2, _>(Shape::new(embedding_shape_as_slice));

    let targets_tensor: Tensor<MyBackend, 1> = Tensor::<MyBackend, 1>::from(targets.as_slice());
    let model =
        ModelConfig::new(embedding_cats, 3, &[64, 32], 0.4).init::<MyBackend>(&device.clone());

    let output = model.forward(embedding_tensor, continuous_tensor);
    println!("Output: {:?}", output);
    //println!("Model: {:?}", model);
}
