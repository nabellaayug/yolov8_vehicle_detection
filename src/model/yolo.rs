use burn::prelude::*;
use crate::model::{Backbone, Neck, DetectionHead};

#[derive(Module, Debug)]
pub struct YOLOv8Nano<B: Backend> {
    backbone: Backbone<B>,
    neck: Neck<B>,
    head: DetectionHead<B>,
}

impl<B: Backend> YOLOv8Nano<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        Self {
            backbone: Backbone::<B>::new(device),
            neck: Neck::new(device),
            head: DetectionHead::new(device, num_classes),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // Backbone
        let (p2, p3, p4) = self.backbone.forward(x);
        
        // Neck
        let (p2_neck, p3_neck, p4_neck) = self.neck.forward(p2, p3, p4);
        
        // Head
        let (pred_p2, pred_p3, pred_p4) = self.head.forward(p2_neck, p3_neck, p4_neck);
        
        (pred_p2, pred_p3, pred_p4)
    }

    pub fn num_classes(&self) -> usize {
        self.head.num_classes()
    }
}