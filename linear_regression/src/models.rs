mod embedding_model;
mod linear_model;
use burn::nn::loss::MseLoss;
use burn::nn::loss::Reduction;
use burn::prelude::*;
use burn::train::RegressionOutput;
use embedding_model::{TaxifareEmbeddingLayerConfig, TaxifareEmbeddingModel};
use itertools::Itertools;
use linear_model::{TaxifareLinearLayerConfig, TaxifareLinearLayerModel};

pub struct ModelConfig {
    embedding_config: TaxifareEmbeddingLayerConfig,
    linear_layers_config: Vec<TaxifareLinearLayerConfig>,
}

impl ModelConfig {
    pub fn new(
        embedding_sizes: Vec<(usize, usize)>,
        n_cont: usize,
        layers: &[usize],
        dropout_rate: f64,
    ) -> Self {
        let mut layer_configuration = layers.to_vec();
        layer_configuration.push(1);
        layer_configuration.insert(0, n_cont + embedding_sizes.len());
        Self {
            embedding_config: TaxifareEmbeddingLayerConfig::new(embedding_sizes, dropout_rate),
            linear_layers_config: layer_configuration
                .iter()
                .tuple_windows()
                .map(|(input, output)| {
                    TaxifareLinearLayerConfig::new(*input, *output, dropout_rate)
                })
                .collect(),
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
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    embedding: TaxifareEmbeddingModel<B>,
    linear_layers: Vec<TaxifareLinearLayerModel<B>>,
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, cat_input: Tensor<B, 2, Int>, cont_input: Tensor<B, 2>) -> Tensor<B, 2> {
        let cat_output = self.embedding.forward(cat_input);
        let mut x = cont_input;
        for layer in &self.linear_layers {
            x = layer.forward(x);
        }
        x
    }

    pub fn forward_regression(
        &self,
        cat_input: Tensor<B, 2, Int>,
        cont_input: Tensor<B, 2>,
        targets: Tensor<B, 2, Float>,
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
