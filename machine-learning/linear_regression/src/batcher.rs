use burn::{data::dataloader::batcher::Batcher, prelude::*};

use super::dataset::mapped_dataset::TaxifareDatasetMappedItem;

#[derive(Clone, Default, Debug)]
pub struct TaxifareBatcher;

#[derive(Clone, Debug)]
pub struct TaxifareBatch<B: Backend> {
    pub cont_features: Tensor<B, 3>,
    pub cat_features: Vec<Tensor<B, 2, Int>>,
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
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 6, 1]))
            .collect();
        let cat_weekday = items
            .iter()
            .map(|item| TensorData::from([item.discrete_weekday]).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();
        let cat_weekday = Tensor::cat(cat_weekday, 1);

        let cat_hour = items
            .iter()
            .map(|item| TensorData::from([item.discrete_hour]).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();
        let cat_hour = Tensor::cat(cat_hour, 1);

        let cat_am_or_pm = items
            .iter()
            .map(|item| TensorData::from([item.discrete_am_or_pm]).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();
        let cat_am_or_pm = Tensor::cat(cat_am_or_pm, 1);

        let predictions = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_data([item.label], device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();

        let cont_features = Tensor::cat(cont_features, 2);
        let cat_features = vec![cat_weekday, cat_hour, cat_am_or_pm];
        let predictions = Tensor::cat(predictions, 1);
        TaxifareBatch {
            cont_features,
            cat_features,
            predictions,
        }
    }
}
