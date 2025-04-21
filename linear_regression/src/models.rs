mod embedding_model;
mod linear_model;
use burn::config::Config;
use burn::nn::loss::MseLoss;
use burn::nn::{BatchNorm, Dropout, DropoutConfig, Embedding, Linear, Relu};
use burn::prelude::*;
use burn::train::RegressionOutput;
use embedding_model::{TaxifareEmbeddingLayerConfig, TaxifareEmbeddingModel};
use linear_model::{TaxifareLinearLayerConfig, TaxifareLinearLayerModel};

#[derive(Config)]
pub struct ModelConfig {
    embedding_config: TaxifareEmbeddingLayerConfig,
    continuous_features: usize,
    linear_layers_config: Vec<TaxifareLinearLayerConfig>,
}

impl ModelConfig {
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
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x)
    }

    pub fn forward_regression(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let output = self.forward(images);
        let loss = MseLoss::forward(&MseLoss::new(), output.clone(), targets.clone());

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}
