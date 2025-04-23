use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig},
    prelude::Backend,
    tensor::{Int, Tensor},
};

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
    pub fn new(embedding_sizes: Vec<(usize, usize)>, dropout_rate: f64) -> Self {
        Self {
            embedding_configs: embedding_sizes
                .iter()
                .map(|(num_features, num_embeddings)| {
                    EmbeddingConfig::new(*num_features, *num_embeddings)
                })
                .collect(),
            dropout: DropoutConfig::new(dropout_rate),
        }
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
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let input_shape = input.shape();
        let rows = input_shape.dims[0];
        let x = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, embed)| embed.forward(input.clone().slice([0..rows, i..(i + 1)])))
            .collect::<Vec<Tensor<B, 3>>>();
        let raw_output = Tensor::stack(x, 2);
        self.dropout_layer.forward(raw_output)
    }
}
