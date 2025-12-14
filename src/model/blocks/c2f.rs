use burn::prelude::*;
use crate::model::blocks::Conv;

#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    cv3: Conv<B>,
}

impl<B: Backend> Bottleneck<B> {
    pub fn new(device: &B::Device, in_channels: usize, out_channels: usize) -> Self {
        let hidden = out_channels / 2;
        
        Self {
            cv1: Conv::new(device, in_channels, hidden, 1, 1),
            cv2: Conv::new(device, hidden, out_channels, 3, 1),
            cv3: Conv::new(device, in_channels, out_channels, 1, 1),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let branch1 = self.cv1.forward(x.clone());
        let branch1 = self.cv2.forward(branch1);
        
        let branch2 = self.cv3.forward(x);
        
        branch1 + branch2
    }
}

#[derive(Module, Debug)]
pub struct C2f<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    bottlenecks: Vec<Bottleneck<B>>,
}

impl<B: Backend> C2f<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        n: usize,
        use_downsample: bool,
    ) -> Self {
        let stride = if use_downsample { 2 } else { 1 };
        
        Self {
            cv1: Conv::new(device, in_channels, out_channels, 3, stride),
            cv2: Conv::new(device, out_channels * 2, out_channels, 1, 1),
            bottlenecks: (0..n)
                .map(|_| Bottleneck::new(device, out_channels, out_channels))
                .collect(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.cv1.forward(x);
        
        let mut y = x.clone();
        for bottleneck in &self.bottlenecks {
            y = bottleneck.forward(y);
        }
        
        let [b, c, h, w] = x.dims();
        let x_flat = x.reshape([b, c / 2, 2, h, w]);
        let y_flat = y.reshape([b, c / 2, 2, h, w]);
        
        // Concatenate (simplified)
        let combined = x_flat; // TODO: proper concat
        let [b, c, _, h, w] = combined.dims();
        let combined = combined.reshape([b, c * 2, h, w]);
        
        self.cv2.forward(combined)
    }
}