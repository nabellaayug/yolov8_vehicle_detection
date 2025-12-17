use super::{Bottleneck, Conv};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct C2f<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
    bottlenecks: Vec<Bottleneck<B>>,
    split_channels: usize,
}

impl<B: Backend> C2f<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        n: usize,
        shortcut: bool,
    ) -> Self {
        let hidden_channels = out_channels / 2;
        let e = 0.5;
        let mut bottlenecks = Vec::new();
        for _ in 0..n {
            bottlenecks.push(Bottleneck::new(
                device,
                hidden_channels,
                hidden_channels,
                shortcut,
                e,
            ));
        }

        Self {
            cv1: Conv::new(device, in_channels, hidden_channels * 2, 1, 1),
            cv2: Conv::new(device, hidden_channels * (2 + n), out_channels, 1, 1),
            bottlenecks,
            split_channels: hidden_channels,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Initial convolution and split
        let x = self.cv1.forward(x);

        // Get dimensions
        let [batch, channels, height, width] = x.dims();

        // Branch 1: direct passthrough (first half of channels)
        let branch1 = x
            .clone()
            .slice([0..batch, 0..self.split_channels, 0..height, 0..width]);

        // Branch 2: goes through bottlenecks (second half of channels)
        let mut branch2 = x.slice([0..batch, self.split_channels..channels, 0..height, 0..width]);

        // Collect all bottleneck outputs for concatenation
        let mut outputs = vec![branch1, branch2.clone()];

        // Apply bottleneck blocks sequentially
        for bottleneck in &self.bottlenecks {
            branch2 = bottleneck.forward(branch2);
            outputs.push(branch2.clone());
        }

        // Concatenate all branches along channel dimension
        let concatenated = Tensor::cat(outputs, 1);

        // Final convolution
        self.cv2.forward(concatenated)
    }
}
