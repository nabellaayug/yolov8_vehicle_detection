use burn::prelude::*;
use crate::model::blocks::Conv;

#[derive(Module, Debug)]
pub struct C2f<B: Backend> {
    cv1: Conv<B>,
    cv2: Conv<B>,
}

impl<B: Backend> C2f<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        _n: usize,
        use_downsample: bool,
    ) -> Self {
        let stride = if use_downsample { 2 } else { 1 };
        
        Self {
            // cv1: in_channels → out_channels (with stride)
            cv1: Conv::new(device, in_channels, out_channels, 3, stride),
            // cv2: out_channels*2 → out_channels (for concat)
            cv2: Conv::new(device, out_channels * 2, out_channels, 1, 1),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // ✅ Branch 1: cv1
        let branch1 = self.cv1.forward(x);
        
        // ✅ Branch 2: same as branch1 (duplicate for concat)
        let branch2 = branch1.clone();
        
        // ✅ Concatenate along channel dimension
        // [batch, out_channels, h, w] + [batch, out_channels, h, w] 
        // = [batch, out_channels*2, h, w]
        let combined = Tensor::cat(vec![branch1, branch2], 1);
        
        // ✅ Apply cv2: out_channels*2 → out_channels
        self.cv2.forward(combined)
    }
}