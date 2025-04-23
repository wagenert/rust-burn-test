mod embedding_model;
mod linear_model;
use burn::nn::loss::MseLoss;
use burn::prelude::*;
use burn::train::RegressionOutput;
use burn::{config::Config, nn::loss::Reduction};
use embedding_model::{TaxifareEmbeddingLayerConfig, TaxifareEmbeddingModel};
use linear_model::{TaxifareLinearLayerConfig, TaxifareLinearLayerModel};

#[derive(Config)]
pub struct ModelConfig {
    embedding_config: TaxifareEmbeddingLayerConfig,
    continuous_features: usize,
    linear_layers_config: Vec<TaxifareLinearLayerConfig>,
}

impl ModelConfig {
    /*
    pub fn new(
        embedding_sizes: Vec<(usize, usize)>,
        n_cont: usize,
        layers: &[usize],
        dropout_rate: f64,
    ) -> Self {
        Self {
            embedding_config: TaxifareEmbeddingLayerConfig::new(embedding_sizes, dropout_rate),
            continuous_features: n_cont,
            linear_layers_config: todo!(),
        }
    }
    */

    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            embedding: self.embedding_config.init(device),
            linear_layers: self
                .linear_layers_config
                .iter()
                .map(|config| config.init(device).unwrap())
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
        let mut cat_output = self.embedding.forward(cat_input);
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
