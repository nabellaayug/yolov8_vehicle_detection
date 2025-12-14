use burn::prelude::*;
use crate::model::blocks::Conv;
use crate::model::blocks::Upsample2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;

#[derive(Module, Debug)]
pub struct Neck<B: Backend> {
    upsample: Upsample2d,
    conv_reduce1: Conv<B>,
    conv_reduce2: Conv<B>,
    conv_fpn1: Conv<B>,
    conv_fpn2: Conv<B>,
    conv_fpn3: Conv<B>,
}

impl<B: Backend> Neck<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            upsample: Upsample2d::new(2),
            conv_reduce1: Conv::<B>::new(device, 512, 256, 1, 1),
            conv_reduce2: Conv::<B>::new(device, 256, 128, 1, 1),
            conv_fpn1: Conv::<B>::new(device, 256, 256, 3, 1),
            conv_fpn2: Conv::<B>::new(device, 256, 256, 3, 1),
            conv_fpn3: Conv::<B>::new(device, 256, 256, 3, 1),
        }
    }

    pub fn forward(
        &self,
        p2: Tensor<B, 4>,
        p3: Tensor<B, 4>,
        p4: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // Top-down
        let p4_up = self.upsample.forward(p4.clone());
        let p4_up = self.conv_reduce1.forward(p4_up);
        let p3_fuse = p3 + p4_up;
        let p3_fuse = self.conv_fpn1.forward(p3_fuse);
        
        let p3_up = self.upsample.forward(p3_fuse.clone());
        let p3_up = self.conv_reduce2.forward(p3_up);
        let p2_fuse = p2 + p3_up;
        let p2_fuse = self.conv_fpn2.forward(p2_fuse);
        
        // Bottom-up
        let p3_down: Conv2d<B>  = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(&p2_fuse.device());
        let p3_down = self.conv_fpn3.forward(p2_fuse.clone());
        let p3_final = p3_fuse + p3_down;
        
        (p2_fuse, p3_final, p4)
    }
}