use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};

use crate::{
    batcher::TaxifareBatcher, dataset::taxifare_dataset::TaxifareDatasetBuilder,
    models::taxifare_model::ModelConfig,
};

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 1)]
    pub num_epochs: usize,

    #[config(default = 1)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 256)]
    pub batch_size: usize,

    #[config(default = 1.0e-4)]
    pub learning_rate: f64,

    pub model: ModelConfig,
    pub optimizer: AdamConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config can not be saved.");

    B::seed(config.seed);

    let dataset_builder = TaxifareDatasetBuilder::new("../TaxiFaresPrepared.csv", 75, Some(42));
    let test_dataset = dataset_builder.test();
    let train_dataset = dataset_builder.train();

    let batcher = TaxifareBatcher;

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model can not be saved.");
}
