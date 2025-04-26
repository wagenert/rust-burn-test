use std::fs::File;

use data_preparation::data::create_input_dataset;
use polars::prelude::*;

fn main() {
    let df = create_input_dataset("NYCTaxiFares.csv").unwrap();
    let mut write_df = df
        .lazy()
        .select([
            col("fare_amount"),
            col("pickup_latitude"),
            col("pickup_longitude"),
            col("dropoff_latitude"),
            col("dropoff_longitude"),
            col("passenger_count").cast(DataType::Float64),
            col("distance"),
            col("pickup_hour"),
            col("pickup_weekday"),
            col("am_or_pm"),
        ])
        .collect()
        .unwrap();
    let mut file = File::create("../TaxiFaresPrepared.csv").unwrap();
    let _csv_writer = CsvWriter::new(&mut file).finish(&mut write_df);
}
