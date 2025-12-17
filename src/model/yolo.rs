use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use super::backbone::Backbone;
use super::neck::Neck;
use super::head::DetectionHead;
use super::blocks::BBoxDecoder;
use super::nms::{Detection, NMS};

#[derive(Module, Debug)]
pub struct YOLOv8<B: Backend> {
    pub backbone: Backbone<B>,
    pub neck: Neck<B>,
    pub head: DetectionHead<B>,
}

impl<B: Backend> YOLOv8<B> {
    pub fn new(device: &B::Device, num_classes: usize, reg_max: usize) -> Self {
        Self {
            backbone: Backbone::new(device),
            neck: Neck::new(device),
            head: DetectionHead::new(device, num_classes, reg_max),
        }
    }

    /// Forward pass - returns raw predictions for training
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
    ) -> (
        (Tensor<B, 4>, Tensor<B, 4>),  // P2: (reg, cls)
        (Tensor<B, 4>, Tensor<B, 4>),  // P3: (reg, cls)
        (Tensor<B, 4>, Tensor<B, 4>),  // P4: (reg, cls)
    ) {
        let (p3, p4, p5) = self.backbone.forward(x);
        let (n3, n4, n5) = self.neck.forward(p3, p4, p5);
        self.head.forward(n3, n4, n5)
    }

    /// Inference with NMS - returns final detections
    pub fn predict(
        &self,
        x: Tensor<B, 4>,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Vec<Detection> {
        let ((reg_p2, cls_p2), (reg_p3, cls_p3), (reg_p4, cls_p4)) = self.forward(x);

        let decoder = BBoxDecoder::new(self.head.reg_max);
        let mut all_detections = Vec::new();

        // Decode P2 (stride 8)
        all_detections.extend(self.decode_scale(
            reg_p2,
            cls_p2,
            &decoder,
            8,
            conf_threshold,
        ));

        // Decode P3 (stride 16)
        all_detections.extend(self.decode_scale(
            reg_p3,
            cls_p3,
            &decoder,
            16,
            conf_threshold,
        ));

        // Decode P4 (stride 32)
        all_detections.extend(self.decode_scale(
            reg_p4,
            cls_p4,
            &decoder,
            32,
            conf_threshold,
        ));

        // Apply NMS across all scales
        NMS::apply(all_detections, iou_threshold, conf_threshold)
    }

    fn decode_scale(
        &self,
        reg: Tensor<B, 4>,
        cls: Tensor<B, 4>,
        decoder: &BBoxDecoder,
        stride: usize,
        conf_threshold: f32,
    ) -> Vec<Detection> {
        // Decode bounding boxes
        let boxes = decoder.decode(reg, stride);  // [B, HW, 4]
        
        // Apply sigmoid to classification logits
        let cls = sigmoid(cls);  // [B, num_classes, H, W]

        // Move to CPU for processing
        let box_data: Vec<f32> = boxes.into_data().convert::<f32>().to_vec().unwrap();
        let cls_data: Vec<f32> = cls.clone().into_data().convert::<f32>().to_vec().unwrap();

        let [_, num_classes, h, w] = cls.dims();
        let hw = h * w;
        
        let mut detections = Vec::new();

        // Process each spatial location
        for i in 0..hw {
            // Find best class
            let mut best_class = 0;
            let mut best_score = 0.0f32;

            for c in 0..num_classes {
                let score = cls_data[c * hw + i];
                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            // Filter by confidence
            if best_score < conf_threshold {
                continue;
            }

            // Get bbox coordinates (x1, y1, x2, y2)
            let idx = i * 4;
            let x1 = box_data[idx];
            let y1 = box_data[idx + 1];
            let x2 = box_data[idx + 2];
            let y2 = box_data[idx + 3];

            // Convert to center format (x, y, w, h)
            let cx = (x1 + x2) * 0.5;
            let cy = (y1 + y2) * 0.5;
            let w = x2 - x1;
            let h = y2 - y1;

            detections.push(Detection {
                x: cx,
                y: cy,
                w,
                h,
                confidence: best_score,
                class_id: best_class,
            });
        }

        detections
    }
}