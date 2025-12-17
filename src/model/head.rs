use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct DetectionHead<B: Backend> {
    pub num_classes: usize,
    pub reg_max: usize,
    reg_p2: Conv2d<B>,
    reg_p3: Conv2d<B>,
    reg_p4: Conv2d<B>,
    cls_p2: Conv2d<B>,
    cls_p3: Conv2d<B>,
    cls_p4: Conv2d<B>,
}

impl<B: Backend> DetectionHead<B> {
    pub fn new(device: &B::Device, num_classes: usize, reg_max: usize) -> Self {
        let reg_ch = 4 * (reg_max + 1);
        Self {
            num_classes,
            reg_max,
            reg_p2: Conv2dConfig::new([128, reg_ch], [1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0))
                .with_bias(true)
                .init(device),
            reg_p3: Conv2dConfig::new([256, reg_ch], [1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0))
                .with_bias(true)
                .init(device),
            reg_p4: Conv2dConfig::new([512, reg_ch], [1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0))
                .with_bias(true)
                .init(device),
            cls_p2: Conv2dConfig::new([128, num_classes], [1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0))
                .with_bias(true)
                .init(device),
            cls_p3: Conv2dConfig::new([256, num_classes], [1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0))
                .with_bias(true)
                .init(device),
            cls_p4: Conv2dConfig::new([512, num_classes], [1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0))
                .with_bias(true)
                .init(device),
        }
    }

    pub fn forward(
        &self,
        p2: Tensor<B, 4>,
        p3: Tensor<B, 4>,
        p4: Tensor<B, 4>,
    ) -> (
        (Tensor<B, 4>, Tensor<B, 4>),
        (Tensor<B, 4>, Tensor<B, 4>),
        (Tensor<B, 4>, Tensor<B, 4>),
    ) {
        (
            (self.reg_p2.forward(p2.clone()), self.cls_p2.forward(p2)),
            (self.reg_p3.forward(p3.clone()), self.cls_p3.forward(p3)),
            (self.reg_p4.forward(p4.clone()), self.cls_p4.forward(p4)),
        )
    }
}