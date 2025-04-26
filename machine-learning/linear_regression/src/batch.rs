use burn::{data::dataloader::batcher::Batcher, prelude::*};

struct TrainingBatcher {}

struct TrainingBatch<B: Backend> {
    cont_input: Tensor<B, 3>,
    cat_input: Vec<Tensor<B, 2, Int>>,
    target_output: Tensor<B, 1>,
}

impl<B: Backend> Batcher<B, (Vec<Vec<f64>>, Vec<Vec<i32>>), TrainingBatch<B>> for TrainingBatcher {
    fn batch(
        &self,
        items: Vec<(Vec<Vec<f64>>, Vec<Vec<i32>>)>,
        device: &<B as Backend>::Device,
    ) -> TrainingBatch<B> {
        todo!()
    }
}
