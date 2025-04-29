use crate::batcher::TaxifareBatch;

use super::embedding_model::{TaxifareEmbeddingLayerConfig, TaxifareEmbeddingModel};
use super::linear_model::{TaxifareLinearLayerConfig, TaxifareLinearLayerModel};
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::loss::MseLoss;
use burn::nn::loss::Reduction;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    embedding_config: TaxifareEmbeddingLayerConfig,
    linear_layers_config: Vec<TaxifareLinearLayerConfig>,
    output_layer_config: LinearConfig,
    cont_norm_layer_config: BatchNormConfig,
}

impl ModelConfig {
    pub fn new(
        embedding_sizes: Vec<(usize, usize)>,
        n_cont: usize,
        layers: &[usize],
        dropout_rate: f64,
    ) -> Self {
        let mut layer_configuration = layers.to_vec();
        layer_configuration.insert(
            0,
            n_cont + embedding_sizes.iter().map(|ebsz| ebsz.1).sum::<usize>(),
        );
        println!("Resulting layer configuration{layer_configuration:?}");
        Self {
            embedding_config: TaxifareEmbeddingLayerConfig::new(embedding_sizes, dropout_rate),
            linear_layers_config: layer_configuration
                .iter()
                .tuple_windows()
                .map(|(input, output)| {
                    TaxifareLinearLayerConfig::new(*input, *output, dropout_rate)
                })
                .collect(),
            output_layer_config: LinearConfig::new(*layers.last().unwrap(), 1),
            cont_norm_layer_config: BatchNormConfig::new(n_cont),
        }
    }

    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            embedding: self.embedding_config.init(device),
            linear_layers: self
                .linear_layers_config
                .iter()
                .map(|config| config.init(device))
                .collect(),
            output_layer: self.output_layer_config.init::<B>(device),
            cont_input_norm_layer: self.cont_norm_layer_config.init::<B, 1>(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    embedding: TaxifareEmbeddingModel<B>,
    linear_layers: Vec<TaxifareLinearLayerModel<B>>,
    cont_input_norm_layer: BatchNorm<B, 1>,
    output_layer: Linear<B>,
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(
        &self,
        cat_input: Vec<Tensor<B, 2, Int>>,
        cont_input: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let cat_output = self.embedding.forward(cat_input);
        let cont_norm_input = self.cont_input_norm_layer.forward(cont_input);
        let mut x = Tensor::cat(vec![cont_norm_input, cat_output], 1);
        for layer in &self.linear_layers {
            x = layer.forward(x);
        }
        x = self.output_layer.forward(x);
        x.squeeze(0)
    }

    pub fn forward_regression(
        &self,
        cat_input: Vec<Tensor<B, 2, Int>>,
        cont_input: Tensor<B, 3>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let output = self.forward(cat_input, cont_input);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<TaxifareBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: TaxifareBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(item.cat_features, item.cont_features, item.predictions);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TaxifareBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: TaxifareBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(item.cat_features, item.cont_features, item.predictions)
    }
}
