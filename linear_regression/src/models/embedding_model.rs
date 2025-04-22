use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig},
    prelude::Backend,
    tensor::{Int, Tensor},
};

#[derive(Config)]
pub struct TaxifareEmbeddingLayerConfig {
    embedding_configs: Vec<EmbeddingConfig>,
    dropout: DropoutConfig,
}

impl Default for TaxifareEmbeddingLayerConfig {
    fn default() -> Self {
        Self {
            embedding_configs: Vec::new(),
            dropout: DropoutConfig::new(0.5),
        }
    }
}
impl TaxifareEmbeddingLayerConfig {
    pub fn with_embedding_configs(mut self, embedding_sizes: &[(usize, usize)]) -> Self {
        self.embedding_configs = embedding_sizes
            .iter()
            .map(|(num_features, num_embeddings)| {
                EmbeddingConfig::new(*num_features, *num_embeddings)
            })
            .collect();
        self
    }

    pub fn with_dropout_rate(mut self, dropout_rate: f64) -> Self {
        self.dropout = DropoutConfig::new(dropout_rate);
        self
    }

    pub fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> TaxifareEmbeddingModel<B> {
        TaxifareEmbeddingModel {
            embeddings: self
                .embedding_configs
                .iter()
                .map(|embedding_config| embedding_config.init(device))
                .collect(),
            dropout_layer: self.dropout.init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct TaxifareEmbeddingModel<B: Backend> {
    embeddings: Vec<Embedding<B>>,
    dropout_layer: Dropout,
}

impl<B: Backend> TaxifareEmbeddingModel<B> {
    fn forward(&self, embedding_data: Tensor<B, 3, Int>) -> Tensor<B, 3, Int> {
        let mut x = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, embed)| embed.forward(embedding_data[i]))
            .collect();
        self.dropout_layer.forward(x)
    }
}
