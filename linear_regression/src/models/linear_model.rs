use burn::{
    module::Module,
    nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::Tensor,
};

pub struct TaxifareLinearLayerConfig {
    linear_config: LinearConfig,
    dropout_config: DropoutConfig,
    norm_config: BatchNormConfig,
}

impl TaxifareLinearLayerConfig {
    pub fn new(inputs: usize, outputs: usize, dropout_rate: f64) -> Self {
        Self {
            linear_config: LinearConfig::new(inputs, outputs),
            dropout_config: DropoutConfig::new(dropout_rate),
            norm_config: BatchNormConfig::new(outputs),
        }
    }

    pub fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> TaxifareLinearLayerModel<B> {
        TaxifareLinearLayerModel {
            linear_layer: self.linear_config.init(device),
            dropout_layer: self.dropout_config.init(),
            norm_layer: self.norm_config.init(device),
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct TaxifareLinearLayerModel<B: Backend> {
    linear_layer: Linear<B>,
    dropout_layer: Dropout,
    norm_layer: BatchNorm<B, 1>,
    activation: Relu,
}

impl<B: Backend> TaxifareLinearLayerModel<B> {
    pub fn forward(&self, linear_data: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = self.linear_layer.forward(linear_data);
        x = self.activation.forward(x);
        let input_norm_dims: [usize; 2] = x.shape().dims();
        let norm_tensor: Tensor<B, 3> = x.clone().unsqueeze_dims(&[
            1,
            input_norm_dims[0] as isize,
            input_norm_dims[1] as isize,
        ]);
        //let norm_tensor = x.clone().unsqueeze();
        let norm_output_tensor = self.norm_layer.forward::<3>(norm_tensor);
        println!("Batch Norm Output Shape {:?}", norm_output_tensor.shape());
        self.dropout_layer.forward(x)
    }
}
