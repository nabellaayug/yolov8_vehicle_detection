use burn::prelude::*;
use crate::model::blocks::{Conv, C2f, Upsample2d};

#[derive(Module, Debug)]
pub struct Neck<B: Backend> {
    // FPN (Top-Down)
    upsample1: Upsample2d,
    reduce1: Conv<B>,              // Reduce P5 channels before concat
    reduce_p4: Conv<B>,            // Reduce P4 channels before concat
    c2f_fpn1: C2f<B>,              // After concat P5+P4
    
    upsample2: Upsample2d,
    reduce2: Conv<B>,              // Reduce P4 channels before concat
    reduce_p3: Conv<B>,            // Reduce P3 channels before concat
    c2f_fpn2: C2f<B>,              // After concat P4+P3
    
    // PAN (Bottom-Up)
    conv_pan1: Conv<B>,            // Downsample P3
    c2f_pan1: C2f<B>,              // After concat P3_down+P4
    
    conv_pan2: Conv<B>,            // Downsample P4
    c2f_pan2: C2f<B>,              // After concat P4_down+P5
}

impl<B: Backend> Neck<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // FPN
            upsample1: Upsample2d::new(2),
            reduce1: Conv::new(device, 512, 256, 1, 1),      // P5: 512 -> 256
            reduce_p4: Conv::new(device, 512, 256, 1, 1),    // P4: 512 -> 256
            c2f_fpn1: C2f::new(device, 512, 256, 3, true),  // 512 (256+256) -> 256
            
            upsample2: Upsample2d::new(2),
            reduce2: Conv::new(device, 256, 128, 1, 1),      // P4: 256 -> 128
            reduce_p3: Conv::new(device, 256, 128, 1, 1),    // P3: 256 -> 128
            c2f_fpn2: C2f::new(device, 256, 128, 3, true),  // 256 (128+128) -> 128
            
            // PAN
            conv_pan1: Conv::new(device, 128, 128, 3, 2),    // P3: stride 2
            c2f_pan1: C2f::new(device, 384, 256, 3, true),  //384 (128+256) -> 256
            
            conv_pan2: Conv::new(device, 256, 256, 3, 2),    // P4: stride 2
            c2f_pan2: C2f::new(device, 768, 512, 3, true),  // 768 (256+512) -> 512
        }
    }

    /// Returns (N3, N4, N5) for detection heads
    /// N3: 128 channels (stride 8)
    /// N4: 256 channels (stride 16)
    /// N5: 512 channels (stride 32)
    pub fn forward(
        &self,
        p3: Tensor<B, 4>,   // [B, 256, H/8, W/8]
        p4: Tensor<B, 4>,   // [B, 512, H/16, W/16]
        p5: Tensor<B, 4>,   // [B, 512, H/32, W/32]
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // -------- FPN (Top-Down) --------
        // P5 -> P4
        let p5_up = self.upsample1.forward(p5.clone());     // [B, 512, H/16, W/16]
        let p5_up = self.reduce1.forward(p5_up);            // [B, 256, H/16, W/16]
        let p4_reduced = self.reduce_p4.forward(p4);        // [B, 256, H/16, W/16]
        let p4_cat = Tensor::cat(vec![p5_up, p4_reduced], 1); // [B, 512, H/16, W/16] 
        let p4_fpn = self.c2f_fpn1.forward(p4_cat);         // [B, 256, H/16, W/16]
        
        // P4 -> P3
        let p4_up = self.upsample2.forward(p4_fpn.clone()); // [B, 256, H/8, W/8]
        let p4_up = self.reduce2.forward(p4_up);            // [B, 128, H/8, W/8]
        let p3_reduced = self.reduce_p3.forward(p3);        // [B, 128, H/8, W/8]
        let p3_cat = Tensor::cat(vec![p4_up, p3_reduced], 1); // [B, 256, H/8, W/8] 
        let n3 = self.c2f_fpn2.forward(p3_cat);             // [B, 128, H/8, W/8]
        
        // -------- PAN (Bottom-Up) --------
        // P3 -> P4
        let p3_down = self.conv_pan1.forward(n3.clone());      // [B, 128, H/16, W/16]
        let p4_cat = Tensor::cat(vec![p3_down, p4_fpn], 1);    // [B, 384, H/16, W/16] (128+256)
        let n4 = self.c2f_pan1.forward(p4_cat);                // Need c2f_pan1: 384 -> 256
        
        // P4 -> P5
        let p4_down = self.conv_pan2.forward(n4.clone());      // [B, 256, H/32, W/32]
        let p5_cat = Tensor::cat(vec![p4_down, p5], 1);        // [B, 768, H/32, W/32] (256+512)
        let n5 = self.c2f_pan2.forward(p5_cat);                // Need c2f_pan2: 768 -> 512
        
        (n3, n4, n5)
    }
}