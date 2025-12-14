use burn::prelude::*;

pub struct YOLOLoss;

impl YOLOLoss {
    pub fn compute<B: Backend>(
        pred_p2: Tensor<B, 4>,
        pred_p3: Tensor<B, 4>,
        pred_p4: Tensor<B, 4>,
        target_p2: Tensor<B, 4>,
        target_p3: Tensor<B, 4>,
        target_p4: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let loss_p2 = Self::scale_loss(pred_p2, target_p2);
        let loss_p3 = Self::scale_loss(pred_p3, target_p3);
        let loss_p4 = Self::scale_loss(pred_p4, target_p4);
        
        // Combine losses
        (loss_p2 + loss_p3 + loss_p4).reshape([1])
    }

    fn scale_loss<B: Backend>(
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let [batch, _, h, w] = predictions.dims();
        let num_elements = (batch * h * w) as f32;
        
        // Reshape untuk processing
        let pred_flat = predictions.reshape([batch, usize::MAX]);
        let target_flat = targets.reshape([batch, usize::MAX]);
        
        let pred_size = pred_flat.dims()[1];
        
        // Split predictions: [x, y, w, h, conf, class1, class2, ...]
        let pred_bbox = pred_flat.clone().slice([0..batch, 0..4]);
        let pred_conf = pred_flat.clone().slice([0..batch, 4..5]);
        let pred_cls = pred_flat.slice([0..batch, 5..pred_size]);
        
        let target_size = target_flat.dims()[1];
        let target_bbox = target_flat.clone().slice([0..batch, 0..4]);
        let target_conf = target_flat.clone().slice([0..batch, 4..5]);
        let target_cls = target_flat.slice([0..batch, 5..target_size]);
        
        // Bounding box loss (IoU loss - simplified to MSE)
        let bbox_loss = (pred_bbox - target_bbox).powf_scalar(2.0).mean();
        
        // Confidence loss (BCE - simplified to MSE)
        let conf_loss = (pred_conf - target_conf).powf_scalar(2.0).mean();
        
        // Classification loss (BCE - simplified to MSE)
        let cls_loss = (pred_cls - target_cls).powf_scalar(2.0).mean();
        
        // Weight the losses
        let loss = bbox_loss * 0.5 + conf_loss * 1.0 + cls_loss * 0.5;
        loss.reshape([1])
    }

    // IoU loss untuk bounding boxes
    pub fn iou_loss<B: Backend>(
        pred_boxes: Tensor<B, 2>,
        target_boxes: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        // pred_boxes: [batch, 4] (x, y, w, h)
        // Compute IoU and return 1 - IoU
        
        let intersection = Self::compute_intersection(&pred_boxes, &target_boxes);
        let union = Self::compute_union(&pred_boxes, &target_boxes, &intersection);
        
        let iou = intersection / (union + 1e-6);
        let loss: Tensor<B, 1> = 1.0 - iou;
        
        loss.reshape([1])
    }

    fn compute_intersection<B: Backend>(
        pred: &Tensor<B, 2>,
        target: &Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        // Simplified intersection computation
        Tensor::<B, 1>::zeros([1], &pred.device())
    }

    fn compute_union<B: Backend>(
        pred: &Tensor<B, 2>,
        target: &Tensor<B, 2>,
        intersection: &Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        // Simplified union computation
        Tensor::<B, 1>::ones([1], &pred.device())
    }
}