use polars::prelude::*;

fn main() {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("NYCTaxiFares.csv".into()))
        .expect("Failed to read CSV file")
        .finish()
        .expect("Failed to finish reading CSV")
        .lazy();

    let df = df
        .select([col("pickup_datetime")
            .str()
            .to_datetime(
                Some(TimeUnit::Milliseconds),
                Some("UTC".into()),
                StrptimeOptions {
                    format: Some("%Y-%m-%d %H:%M:%S UTC".into()),
                    ..StrptimeOptions::default()
                },
                lit("raise"),
            )
            .dt()
            .convert_time_zone("America/New_York".into())
            .alias("dt_pickup_datetime")])
        .with_columns([
            col("dt_pickup_datetime").dt().hour().alias("pickup_hour"),
            col("dt_pickup_datetime")
                .dt()
                .weekday()
                .alias("pickup_weekday"),
        ])
        .collect()
        .expect("Failed to collect DataFrame");
    println!("{:?}", df.head(Some(5)));
}
