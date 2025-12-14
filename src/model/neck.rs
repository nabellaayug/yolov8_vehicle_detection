use burn::prelude::*;
use crate::model::blocks::Conv;
use crate::model::blocks::Upsample2d;

#[derive(Module, Debug)]
pub struct Neck<B: Backend> {
    upsample: Upsample2d,
    
    // Top-down pathway
    conv_reduce1: Conv<B>,      // 512 → 256
    conv_reduce2: Conv<B>,      // 256 → 128
    conv_fpn1: Conv<B>,         // 256 → 256 (after fusion)
    conv_fpn2: Conv<B>,         // 128 → 128 (after fusion)
    
    // Bottom-up pathway
    conv_down1: Conv<B>,        // 128 → 256 (downsample)
    conv_fpn3: Conv<B>,         // 256 → 256 (after fusion)
}

impl<B: Backend> Neck<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            upsample: Upsample2d::new(2),
            
            // Top-down
            conv_reduce1: Conv::<B>::new(device, 512, 256, 1, 1),  // 512 → 256
            conv_reduce2: Conv::<B>::new(device, 256, 128, 1, 1),  // 256 → 128
            conv_fpn1: Conv::<B>::new(device, 256, 256, 3, 1),     // 256 → 256
            conv_fpn2: Conv::<B>::new(device, 128, 128, 3, 1),     // ✅ 128 → 128
            
            // Bottom-up
            conv_down1: Conv::<B>::new(device, 128, 256, 3, 2),    // 128 → 256 (stride=2)
            conv_fpn3: Conv::<B>::new(device, 256, 256, 3, 1),     // 256 → 256
        }
    }

    pub fn forward(
        &self,
        p2: Tensor<B, 4>,      // 128 channels, 80x80
        p3: Tensor<B, 4>,      // 256 channels, 40x40
        p4: Tensor<B, 4>,      // 512 channels, 20x20
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        log::info!("Neck forward: p2={}, p3={}, p4={}", 
            p2.dims()[1], p3.dims()[1], p4.dims()[1]);
        
        // ===== TOP-DOWN PATHWAY =====
        // p4 (512ch) → 256ch → fuse with p3
        let p4_up = self.upsample.forward(p4.clone());
        let [b, c, h, w] = p4_up.dims();
        log::debug!("  p4_up: [{}x{}, {}ch]", h, w, c);
        
        let p4_up = self.conv_reduce1.forward(p4_up);              // 512 → 256
        let [b, c, h, w] = p4_up.dims();
        log::debug!("  p4_up after reduce: [{}x{}, {}ch]", h, w, c);
        
        let p3_fuse = p3 + p4_up;                                   // 256 + 256
        let p3_fuse = self.conv_fpn1.forward(p3_fuse);              // 256 → 256
        let [b, c, h, w] = p3_fuse.dims();
        log::debug!("  p3_fuse: [{}x{}, {}ch]", h, w, c);
        
        // p3 (256ch) → 128ch → fuse with p2
        let p3_up = self.upsample.forward(p3_fuse.clone());
        let [b, c, h, w] = p3_up.dims();
        log::debug!("  p3_up: [{}x{}, {}ch]", h, w, c);
        
        let p3_up = self.conv_reduce2.forward(p3_up);              // 256 → 128
        let [b, c, h, w] = p3_up.dims();
        log::debug!("  p3_up after reduce: [{}x{}, {}ch]", h, w, c);
        
        let p2_fuse = p2 + p3_up;                                   // 128 + 128
        let p2_fuse = self.conv_fpn2.forward(p2_fuse);              // ✅ 128 → 128
        let [b, c, h, w] = p2_fuse.dims();
        log::debug!("  p2_fuse: [{}x{}, {}ch]", h, w, c);
        
        // ===== BOTTOM-UP PATHWAY =====
        // p2 (128ch) → 256ch → fuse with p3
        let p2_down = self.conv_down1.forward(p2_fuse.clone());     // 128 → 256 (stride=2)
        let [b, c, h, w] = p2_down.dims();
        log::debug!("  p2_down: [{}x{}, {}ch]", h, w, c);
        
        let p3_final = p3_fuse + p2_down;                           // 256 + 256
        let p3_final = self.conv_fpn3.forward(p3_final);            // 256 → 256
        let [b, c, h, w] = p3_final.dims();
        log::debug!("  p3_final: [{}x{}, {}ch]", h, w, c);
        
        // ===== OUTPUT =====
        // p2_fuse: 128ch, 80x80
        // p3_final: 256ch, 40x40
        // p4: 512ch, 20x20 (unchanged)
        log::info!("Neck output: p2_fuse={}, p3_final={}, p4={}", 
            p2_fuse.dims()[1], p3_final.dims()[1], p4.dims()[1]);
        
        (p2_fuse, p3_final, p4)
    }
}