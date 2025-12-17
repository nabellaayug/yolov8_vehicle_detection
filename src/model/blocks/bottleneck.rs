use burn::prelude::*;
use super::Conv;

#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    add: bool,
}

impl<B: Backend> Bottleneck<B> {
    pub fn new(device: &B::Device, in_channels: usize, out_channels: usize, shortcut: bool, e: f32) -> Self {
        let hidden = (out_channels as f32 * e) as usize;
        
        Self {
            cv1: Conv::new(device, in_channels, hidden, 1, 1),
            cv2: Conv::new(device, hidden, out_channels, 3, 1),
            add: shortcut,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.cv2.forward(self.cv1.forward(x.clone()));

        if self.add{
            y + x 
        } else{
            y
        }
    }
}