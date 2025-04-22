use burn::{
    config::Config,
    module::Module,
    nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::Tensor,
};

enum TaxifareLinearLayerError {
    ConfigurationIncomplete,
}

#[derive(Config)]
pub struct TaxifareLinearLayerConfig {
    linear_config: Option<LinearConfig>,
    dropout_config: DropoutConfig,
    norm_config: Option<BatchNormConfig>,
}

impl Default for TaxifareLinearLayerConfig {
    fn default() -> Self {
        Self {
            linear_config: None,
            dropout_config: DropoutConfig::new(0.5),
            norm_config: None,
        }
    }
}
impl TaxifareLinearLayerConfig {
    pub fn with_linear_config(mut self, inputs: usize, outputs: usize) -> Self {
        self.linear_config = Some(LinearConfig::new(inputs, outputs));
        self.norm_config = Some(BatchNormConfig::new(outputs));
        self
    }

    pub fn with_dropout_rate(mut self, dropout_rate: f64) -> Self {
        self.dropout_config = DropoutConfig::new(dropout_rate);
        self
    }

    pub fn init<B: Backend>(
        &self,
        device: &<B as Backend>::Device,
    ) -> Result<TaxifareLinearLayerModel<B>, TaxifareLinearLayerError> {
        match (self.linear_config.clone(), self.norm_config.clone()) {
            (Some(linear_config), Some(norm_config)) => Ok(TaxifareLinearLayerModel {
                linear_layer: linear_config.init(device),
                dropout_layer: self.dropout_config.init(),
                norm_layer: norm_config.init(device),
                activation: Relu::new(),
            }),
            _ => Err(TaxifareLinearLayerError::ConfigurationIncomplete),
        }
    }
}

#[derive(Module, Debug)]
pub struct TaxifareLinearLayerModel<B: Backend> {
    linear_layer: Linear<B>,
    dropout_layer: Dropout,
    norm_layer: BatchNorm<B, 2>,
    activation: Relu,
}

impl<B: Backend> TaxifareLinearLayerModel<B> {
    pub fn forward(&self, linear_data: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = self.linear_layer.forward(linear_data);
        x = self.activation.forward(x);
        x = self.norm_layer.forward(x);
        self.dropout_layer.forward(x)
    }
}
