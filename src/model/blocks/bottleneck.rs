use burn::prelude::*;
use crate::model::blocks::Conv;

#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    cv3: Conv<B>,
    add: bool,
}

impl<B: Backend> Bottleneck<B> {
    pub fn new(device: &B::Device, in_channels: usize, out_channels: usize, add: bool) -> Self {
        let hidden = out_channels / 2;
        
        Self {
            cv1: Conv::new(device, in_channels, hidden, 1, 1),
            cv2: Conv::new(device, hidden, out_channels, 3, 1),
            cv3: Conv::new(device, in_channels, out_channels, 1, 1),
            add,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let branch1 = self.cv1.forward(x.clone());
        let branch1 = self.cv2.forward(branch1);
        
        if self.add {
            let branch2 = self.cv3.forward(x);
            branch1 + branch2
        } else {
            branch1
        }
    }
}