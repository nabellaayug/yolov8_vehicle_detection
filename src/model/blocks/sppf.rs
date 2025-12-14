use crate::model::blocks::conv::ConvBNSiLU;
use burn::module::Module;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Debug, Module)]
pub struct SPPF<B: Backend> {
    conv1: ConvBNSiLU<B>,
    conv2: ConvBNSiLU<B>,
    pool: MaxPool2d<B>,
    conv3: ConvBNSiLU<B>,
}

impl<B: Backend> SPPF<B> {
    pub fn new(ch_in: usize, ch_out: usize) -> Self {
        let conv1 = ConvBNSiLU::new(ch_in, ch_out, 1, 1, 0);
        let conv2 = ConvBNSiLU::new(ch_out, ch_out, 1, 1, 0);
        let pool = MaxPool2d::new(MaxPool2dConfig::new(5).with_stride(1).with_padding(2));
        let conv3 = ConvBNSiLU::new(ch_out * 4, ch_out, 1, 1, 0);
    }
}

impl<B: Backend>Module<Tensor<B, 4>> for SPPF<B>{
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4>{
        let x = self.conv1.forward(input);
        let x = self.conv2.forward(x.clone());
        let p1 = self.pool.forward(x.clone());
        let p2 = self.pool.forward(p1.clone());
        let p3 = self.pool.forward(p2.clone());
        let cat = Tensor::concat([x, p1, p2, p3], 1);
        self.conv3.forward(cat)
    }
}