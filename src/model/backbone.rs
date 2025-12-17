use burn::prelude::*;
use crate::model::blocks::{Conv, C2f};
use crate::model::blocks::sppf::SPPF;

#[derive(Module, Debug)]
pub struct Backbone<B: Backend> {
    // Stem
    conv1: Conv<B>,
    
    // Stage 1: P1 (stride 2)
    conv2: Conv<B>,
    c2f1: C2f<B>,
    
    // Stage 2: P2 (stride 4)
    conv3: Conv<B>,
    c2f2: C2f<B>,
    
    // Stage 3: P3 (stride 8)
    conv4: Conv<B>,
    c2f3: C2f<B>,
    
    // Stage 4: P4 (stride 16)
    conv5: Conv<B>,
    c2f4: C2f<B>,
    
    // SPPF
    sppf: SPPF<B>,
}

impl<B: Backend> Backbone<B> {
    pub fn new(device: &B::Device) -> Self {
        println!("Creating Backbone...");
        
        println!("Creating conv1...");
        let conv1 = Conv::new(device, 3, 64, 3, 2);
        
        println!("Creating conv2...");
        let conv2 = Conv::new(device, 64, 128, 3, 2);
        
        println!("Creating c2f1...");
        let c2f1 = C2f::new(device, 128, 128, 3, true);
        
        println!("Creating conv3...");
        let conv3 = Conv::new(device, 128, 256, 3, 2);
        
        println!("Creating c2f2...");
        let c2f2 = C2f::new(device, 256, 256, 6, true);
        
        println!("Creating conv4...");
        let conv4 = Conv::new(device, 256, 512, 3, 2);
        
        println!("Creating c2f3...");
        let c2f3 = C2f::new(device, 512, 512, 6, true);
        
        println!("Creating conv5...");
        let conv5 = Conv::new(device, 512, 512, 3, 2);
        
        println!("Creating c2f4...");
        let c2f4 = C2f::new(device, 512, 512, 3, true);
        
        println!("Creating SPPF...");
        let sppf = SPPF::new(device, 512, 512);
        
        println!("Backbone created successfully!");
        
        Self {
            conv1,
            conv2,
            c2f1,
            conv3,
            c2f2,
            conv4,
            c2f3,
            conv5,
            c2f4,
            sppf,
        }
    }

    /// Returns (P3, P4, P5) for multi-scale detection
    /// P3: stride 8, P4: stride 16, P5: stride 32
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // Stem
        let x = self.conv1.forward(x);                  // [B, 64, H/2, W/2]
        
        // Stage 1 (P1)
        let x = self.conv2.forward(x);                  // [B, 128, H/4, W/4]
        let x = self.c2f1.forward(x);
        
        // Stage 2 (P2) - Not used in head but needed for processing
        let x = self.conv3.forward(x);                  // [B, 256, H/8, W/8]
        let p3 = self.c2f2.forward(x);                  // Save P3 for neck
        
        // Stage 3 (P3)
        let x = self.conv4.forward(p3.clone());         // [B, 512, H/16, W/16]
        let p4 = self.c2f3.forward(x);                  // Save P4 for neck
        
        // Stage 4 (P4)
        let x = self.conv5.forward(p4.clone());         // [B, 512, H/32, W/32]
        let x = self.c2f4.forward(x);
        
        // SPPF
        let p5 = self.sppf.forward(x);                  // [B, 512, H/32, W/32]
        
        (p3, p4, p5)
    }
}