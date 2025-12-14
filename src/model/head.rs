use burn::prelude::*;
use crate::model::blocks::Conv;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::Conv2d;

#[derive(Module, Debug)]
pub struct DetectionHead<B: Backend> {
    conv_p2: Conv<B>,
    conv_p3: Conv<B>,
    conv_p4: Conv<B>,
    
    // Prediction heads
    pred_p2: Conv2d<B>,
    pred_p3: Conv2d<B>,
    pred_p4: Conv2d<B>,
    
    num_classes: usize,
}

impl<B: Backend> DetectionHead<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        // ✅ YOLOv8 is anchor-free: 4 (bbox) + num_classes (no objectness)
        let pred_channels = 4 + num_classes;
        
        // ✅ Validation
        assert!(num_classes > 0 && num_classes <= 1000, 
            "num_classes must be between 1 and 1000, got {}", num_classes);
        
        log::info!("DetectionHead init:");
        log::info!("  num_classes = {}", num_classes);
        log::info!("  pred_channels = 4 + {} = {} (anchor-free YOLOv8)", 
            num_classes, pred_channels);
        
        Self {
            conv_p2: Conv::<B>::new(device, 128, 128, 3, 1),
            conv_p3: Conv::<B>::new(device, 256, 256, 3, 1),
            conv_p4: Conv::<B>::new(device, 512, 512, 3, 1),
            
            pred_p2: Conv2dConfig::new([128, pred_channels], [1, 1]).init(device),
            pred_p3: Conv2dConfig::new([256, pred_channels], [1, 1]).init(device),
            pred_p4: Conv2dConfig::new([512, pred_channels], [1, 1]).init(device),
            
            num_classes,
        }
    }

    pub fn forward(
        &self,
        p2: Tensor<B, 4>,
        p3: Tensor<B, 4>,
        p4: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        log::info!("DetectionHead forward:");
        
        let [b, c, h, w] = p2.dims();
        log::info!("  p2 input: shape=[{}, {}, {}, {}]", b, c, h, w);
        let x2 = self.conv_p2.forward(p2);
        let [b, c, h, w] = x2.dims();
        log::info!("  p2 after conv_p2: shape=[{}, {}, {}, {}]", b, c, h, w);
        let pred2 = self.pred_p2.forward(x2);
        let [b, c, h, w] = pred2.dims();
        log::info!("  p2 pred output: shape=[{}, {}, {}, {}]", b, c, h, w);
        
        let [b, c, h, w] = p3.dims();
        log::info!("  p3 input: shape=[{}, {}, {}, {}]", b, c, h, w);
        let x3 = self.conv_p3.forward(p3);
        let [b, c, h, w] = x3.dims();
        log::info!("  p3 after conv_p3: shape=[{}, {}, {}, {}]", b, c, h, w);
        let pred3 = self.pred_p3.forward(x3);
        let [b, c, h, w] = pred3.dims();
        log::info!("  p3 pred output: shape=[{}, {}, {}, {}]", b, c, h, w);
        
        let [b, c, h, w] = p4.dims();
        log::info!("  p4 input: shape=[{}, {}, {}, {}]", b, c, h, w);
        let x4 = self.conv_p4.forward(p4);
        let [b, c, h, w] = x4.dims();
        log::info!("  p4 after conv_p4: shape=[{}, {}, {}, {}]", b, c, h, w);
        let pred4 = self.pred_p4.forward(x4);
        let [b, c, h, w] = pred4.dims();
        log::info!("  p4 pred output: shape=[{}, {}, {}, {}]", b, c, h, w);
        
        (pred2, pred3, pred4)
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}