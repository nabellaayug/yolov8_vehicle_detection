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
        let num_anchors = 3;
        let pred_channels = num_anchors * (5 + num_classes); // x,y,w,h,conf + classes
        
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
        let x2 = self.conv_p2.forward(p2);
        let pred2 = self.pred_p2.forward(x2);
        
        let x3 = self.conv_p3.forward(p3);
        let pred3 = self.pred_p3.forward(x3);
        
        let x4 = self.conv_p4.forward(p4);
        let pred4 = self.pred_p4.forward(x4);
        
        (pred2, pred3, pred4)
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}