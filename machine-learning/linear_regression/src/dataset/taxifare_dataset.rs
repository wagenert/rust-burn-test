use burn::data::dataset::{Dataset, transform::PartialDataset};

use super::{
    mapped_dataset::{
        RawDatafieldToFeaturesMapper, TaxifareDatasetMappedItem, TaxifareMappedDataset,
    },
    raw_dataset::TaxifareRawDatasetBuilder,
};

type TaxifareDataset = PartialDataset<TaxifareMappedDataset, TaxifareDatasetMappedItem>;

pub struct TaxifareDatasetBuilder<'a> {
    file_name: &'a str,
    seed: Option<u64>,
    train_test_split_percentage: usize,
}

impl<'a> TaxifareDatasetBuilder<'a> {
    pub fn new(file_name: &'a str, train_test_split_percentage: usize, seed: Option<u64>) -> Self {
        Self {
            file_name,
            seed,
            train_test_split_percentage,
        }
    }

    pub fn test(&self) -> TaxifareDataset {
        self.init("test").unwrap()
    }

    pub fn train(&self) -> TaxifareDataset {
        self.init("train").unwrap()
    }

    fn init(&self, split: &str) -> Result<TaxifareDataset, std::io::Error> {
        let dataset = TaxifareRawDatasetBuilder::new(self.file_name, self.seed)
            .init()
            .expect("Can not read csv file.");
        let dataset = TaxifareMappedDataset::new(dataset, RawDatafieldToFeaturesMapper);
        let dataset_len = dataset.len();
        let split_idx = dataset_len * self.train_test_split_percentage / 100;
        match split {
            "train" => Ok(TaxifareDataset::new(dataset, 0, split_idx)),
            "test" => Ok(TaxifareDataset::new(dataset, split_idx, dataset_len)),
            _ => panic!("Unknown split: {split}"),
        }
    }
}
