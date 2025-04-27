use burn::data::dataset::{InMemDataset, transform::ShuffledDataset};
use csv::ReaderBuilder;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct TaxifareDatasetRawItem {
    pub(crate) fare_amount: f64,
    pub(crate) pickup_latitude: f64,
    pub(crate) pickup_longitude: f64,
    pub(crate) dropoff_latitude: f64,
    pub(crate) dropoff_longitude: f64,
    pub(crate) passenger_count: f64,
    pub(crate) distance: f64,
    pub(crate) pickup_hour: u8,
    pub(crate) pickup_weekday: u8,
    pub(crate) am_or_pm: u8,
}

type TaxifareInMemDataset = InMemDataset<TaxifareDatasetRawItem>;
pub type TaxifareRawDataset = ShuffledDataset<TaxifareInMemDataset, TaxifareDatasetRawItem>;

pub(crate) struct TaxifareRawDatasetBuilder<'a> {
    file_name: &'a str,
    seed: Option<u64>,
}

impl<'a> TaxifareRawDatasetBuilder<'a> {
    pub fn new(file_name: &'a str, seed: Option<u64>) -> Self {
        Self { file_name, seed }
    }

    pub fn init(&self) -> Result<TaxifareRawDataset, std::io::Error> {
        let mut reader_builder = ReaderBuilder::new();
        reader_builder.has_headers(true);
        let dataset = TaxifareInMemDataset::from_csv(self.file_name, &reader_builder)?;
        if let Some(seed) = self.seed {
            Ok(TaxifareRawDataset::with_seed(dataset, seed))
        } else {
            Ok(TaxifareRawDataset::new(dataset, &mut StdRng::from_os_rng()))
        }
    }
}
