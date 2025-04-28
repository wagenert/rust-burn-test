use burn::{data::dataloader::batcher::Batcher, prelude::*};

use super::dataset::mapped_dataset::TaxifareDatasetMappedItem;

pub struct TaxifareBatcher;

pub struct TaxifareBatch<B: Backend> {
    pub cont_features: Tensor<B, 3>,
    pub cat_features: Tensor<B, 3, Int>,
    pub predictions: Tensor<B, 2>,
}

impl<B: Backend> Batcher<B, TaxifareDatasetMappedItem, TaxifareBatch<B>> for TaxifareBatcher {
    fn batch(
        &self,
        items: Vec<TaxifareDatasetMappedItem>,
        device: &<B as Backend>::Device,
    ) -> TaxifareBatch<B> {
        let cont_features = items
            .iter()
            .map(|item| TensorData::from(item.continuous_features).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 3>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 6, 1]))
            .collect();
        let cat_features = items
            .iter()
            .map(|item| TensorData::from(item.discrete_features).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 3, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 3, 1]))
            .collect();

        let predictions = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_data([item.label], device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();

        let cont_features = Tensor::cat(cont_features, 2);
        let cat_features = Tensor::cat(cat_features, 2);
        let predictions = Tensor::cat(predictions, 1);
        TaxifareBatch {
            cont_features,
            cat_features,
            predictions,
        }
    }
}
