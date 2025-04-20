use itertools::izip;
use polars::prelude::*;

const EARTH_RADIUS: f64 = 6371.0;
///
///Calculates the haversine distance between 2 sets of GPS coordinates in df
///
fn haversine_distance(plat: f64, plong: f64, dlat: f64, dlong: f64) -> f64 {
    let phi1 = plat.to_radians();
    let phi2 = dlat.to_radians();

    let delta_phi = (dlat - plat).to_radians();
    let delta_lambda = (dlong - plong).to_radians();

    let a = (delta_phi / 2.0).sin().powi(2)
        + phi1.cos() * phi2.cos() * (delta_lambda / 2.0).sin().powi(2);
    let c = 2.0 * libm::atan2(a.sqrt(), (1.0 - a).sqrt());
    EARTH_RADIUS * c // in kilometers
}

pub fn create_input_dataset(filename: &str) -> Result<DataFrame, PolarsError> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(filename.into()))
        .expect("Failed to read CSV file")
        .finish()
        .expect("Failed to finish reading CSV")
        .lazy();
    let df_timebased = df
        .clone()
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
        ]);

    let df_distance = df.clone().select([as_struct(vec![
        col("pickup_latitude"),
        col("pickup_longitude"),
        col("dropoff_latitude"),
        col("dropoff_longitude"),
    ])
    .map(
        |s| {
            let ca = s.struct_()?;
            let plat = ca.field_by_name("pickup_latitude")?;
            let plong = ca.field_by_name("pickup_longitude")?;
            let dlat = ca.field_by_name("dropoff_latitude")?;
            let dlong = ca.field_by_name("dropoff_longitude")?;
            let plat = plat.f64()?;
            let plong = plong.f64()?;
            let dlat = dlat.f64()?;
            let dlong = dlong.f64()?;

            let out: Float64Chunked = izip!(
                plat.into_iter(),
                plong.into_iter(),
                dlat.into_iter(),
                dlong.into_iter()
            )
            .map(|(opt_plat, opt_plong, opt_dlat, opt_dlong)| {
                match (opt_plat, opt_plong, opt_dlat, opt_dlong) {
                    (Some(plat), Some(plong), Some(dlat), Some(dlong)) => {
                        Some(haversine_distance(plat, plong, dlat, dlong))
                    }
                    _ => None,
                }
            })
            .collect();

            Ok(Some(out.into_column()))
        },
        GetOutput::from_type(DataType::Float64),
    )
    .alias("distance")]);

    concat_lf_horizontal([df, df_timebased, df_distance], UnionArgs::default())
        .unwrap()
        .collect()
}
