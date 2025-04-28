use burn::data::dataset::transform::{Mapper, MapperDataset};

use super::raw_dataset::{TaxifareDatasetRawItem, TaxifareRawDataset};

#[derive(Clone, Debug)]
pub struct TaxifareDatasetMappedItem {
    pub discrete_features: [u8; 3],
    pub continuous_features: [f64; 6],
    pub label: f64,
}

pub struct RawDatafieldToFeaturesMapper;

impl Mapper<TaxifareDatasetRawItem, TaxifareDatasetMappedItem> for RawDatafieldToFeaturesMapper {
    fn map(&self, item: &TaxifareDatasetRawItem) -> TaxifareDatasetMappedItem {
        TaxifareDatasetMappedItem {
            discrete_features: [item.pickup_weekday, item.pickup_hour, item.am_or_pm],
            continuous_features: [
                item.pickup_latitude,
                item.pickup_longitude,
                item.dropoff_latitude,
                item.dropoff_longitude,
                item.passenger_count,
                item.distance,
            ],
            label: item.fare_amount,
        }
    }
}

pub(crate) type TaxifareMappedDataset =
    MapperDataset<TaxifareRawDataset, RawDatafieldToFeaturesMapper, TaxifareDatasetRawItem>;
