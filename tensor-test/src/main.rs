use burn::backend::Wgpu;
use burn::tensor::Tensor;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    let tensor_3d: Tensor<MyBackend, 2> =
        Tensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
    println!("{tensor_3d:?}");
    println!("{:?}", tensor_3d.shape());
}
