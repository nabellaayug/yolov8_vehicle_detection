use burn::prelude::*;
use crate::model::blocks::{Conv, C2f};

#[derive(Module, Debug)]
pub struct Backbone<B: Backend> {
    stem: Conv<B>,
    stage1: C2f<B>,
    stage2: C2f<B>,
    stage3: C2f<B>,
    stage4: C2f<B>,
}

impl<B: Backend> Backbone<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            stem: Conv::<B>::new(device, 3, 32, 3, 2),      // 640 -> 320
            stage1: C2f::<B>::new(device, 32, 64, 1, true),   // 320 -> 160
            stage2: C2f::<B>::new(device, 64, 128, 2, true),  // 160 -> 80
            stage3: C2f::<B>::new(device, 128, 256, 2, true), // 80 -> 40
            stage4: C2f::<B>::new(device, 256, 512, 1, true), // 40 -> 20
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.stem.forward(x);
        
        let p1 = self.stage1.forward(x);      // 160x160
        let p2 = self.stage2.forward(p1);     // 80x80
        let p3 = self.stage3.forward(p2.clone());     // 40x40
        let p4 = self.stage4.forward(p3.clone());     // 20x20
        
        // Return multi-scale features
        (p2, p3, p4)
    }
}