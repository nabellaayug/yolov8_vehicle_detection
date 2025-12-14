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
            stem: Conv::<B>::new(device, 3, 32, 3, 2),      // 416 -> 208
            stage1: C2f::<B>::new(device, 32, 64, 1, true),   // 208 -> 104
            stage2: C2f::<B>::new(device, 64, 128, 2, true),  // 104 -> 52
            stage3: C2f::<B>::new(device, 128, 256, 2, true), // 52 -> 26
            stage4: C2f::<B>::new(device, 256, 512, 1, true), // 26 -> 13
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [b, c, h, w] = x.dims();
        log::info!("Input: shape=[{}, {}, {}, {}]", b, c, h, w);
        
        let x = self.stem.forward(x);
        let [b, c, h, w] = x.dims();
        log::info!("After stem: shape=[{}, {}, {}, {}]", b, c, h, w);
        
        let p1 = self.stage1.forward(x);
        let [b, c, h, w] = p1.dims();
        log::info!("After stage1 (p1): shape=[{}, {}, {}, {}]", b, c, h, w);
        
        let p2 = self.stage2.forward(p1);
        let [b, c, h, w] = p2.dims();
        log::info!("After stage2 (p2): shape=[{}, {}, {}, {}]", b, c, h, w);
        
        let p3 = self.stage3.forward(p2.clone());
        let [b, c, h, w] = p3.dims();
        log::info!("After stage3 (p3): shape=[{}, {}, {}, {}]", b, c, h, w);
        
        let p4 = self.stage4.forward(p3.clone());
        let [b, c, h, w] = p4.dims();
        log::info!("After stage4 (p4): shape=[{}, {}, {}, {}]", b, c, h, w);
        
        // ✅ Return: p2(128ch,80x80), p3(256ch,40x40), p4(512ch,20x20)
        log::info!("Returning: p2={}, p3={}, p4={}", 
            p2.dims()[1], p3.dims()[1], p4.dims()[1]);
        
        (p2, p3, p4)
    }
}