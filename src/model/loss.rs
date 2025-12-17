use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};

pub struct Yolov8Loss {
    pub num_classes: usize,
    pub reg_max: usize,
    pub box_weight: f32,
    pub cls_weight: f32,
    pub dfl_weight: f32,
}

impl Yolov8Loss {
    pub fn new(
        num_classes: usize,
        reg_max: usize,
        box_weight: f32,
        cls_weight: f32,
        dfl_weight: f32,
    ) -> Self {
        Self {
            num_classes,
            reg_max,
            box_weight,
            cls_weight,
            dfl_weight,
        }
    }

    /// Compute loss for single scale
    pub fn compute_single_scale<B: Backend>(
        &self,
        pred_reg: &Tensor<B, 4>,
        pred_cls: &Tensor<B, 4>,
        target_reg: &Tensor<B, 4>,
        target_cls: &Tensor<B, 4>,
        obj_mask: &Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let dfl = self.dfl_loss(pred_reg, target_reg, obj_mask);
        let cls = self.bce_loss(pred_cls, target_cls, obj_mask);

        (dfl * self.dfl_weight + cls * self.cls_weight).reshape([1])
    }

    /// Compute multi-scale loss
    pub fn compute_multi_scale<B: Backend>(
        &self,
        // P2 predictions and targets
        pred_reg_p2: &Tensor<B, 4>,
        pred_cls_p2: &Tensor<B, 4>,
        target_reg_p2: &Tensor<B, 4>,
        target_cls_p2: &Tensor<B, 4>,
        obj_mask_p2: &Tensor<B, 4>,
        // P3 predictions and targets
        pred_reg_p3: &Tensor<B, 4>,
        pred_cls_p3: &Tensor<B, 4>,
        target_reg_p3: &Tensor<B, 4>,
        target_cls_p3: &Tensor<B, 4>,
        obj_mask_p3: &Tensor<B, 4>,
        // P4 predictions and targets
        pred_reg_p4: &Tensor<B, 4>,
        pred_cls_p4: &Tensor<B, 4>,
        target_reg_p4: &Tensor<B, 4>,
        target_cls_p4: &Tensor<B, 4>,
        obj_mask_p4: &Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let loss_p2 = self.compute_single_scale(
            pred_reg_p2,
            pred_cls_p2,
            target_reg_p2,
            target_cls_p2,
            obj_mask_p2,
        );

        let loss_p3 = self.compute_single_scale(
            pred_reg_p3,
            pred_cls_p3,
            target_reg_p3,
            target_cls_p3,
            obj_mask_p3,
        );

        let loss_p4 = self.compute_single_scale(
            pred_reg_p4,
            pred_cls_p4,
            target_reg_p4,
            target_cls_p4,
            obj_mask_p4,
        );

        (loss_p2 + loss_p3 + loss_p4) / 3.0
    }

    /// Distribution Focal Loss for bbox regression
    fn dfl_loss<B: Backend>(
        &self,
        pred: &Tensor<B, 4>,
        target: &Tensor<B, 4>,
        obj_mask: &Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let bins = self.reg_max + 1;
        let [b, _, h, w] = pred.dims();

        // Reshape to [B, 4, bins, H, W]
        let pred = pred.clone().reshape([b, 4, bins, h, w]);
        let pred = softmax(pred, 2);

        let target = target.clone().reshape([b, 4, bins, h, w]);

        // Expand obj_mask to match shape
        let obj_mask = obj_mask
            .clone()
            .repeat_dim(1, 4 * bins)
            .reshape([b, 4, bins, h, w]);

        // Cross-entropy loss with mask
        let loss = -(target * pred.log()) * obj_mask.clone();

        // Normalize by number of objects
        let num_objects = obj_mask.sum().clamp_min(1.0);
        loss.sum() / num_objects
    }

    /// Binary Cross-Entropy loss for classification
    fn bce_loss<B: Backend>(
        &self,
        pred: &Tensor<B, 4>,
        target: &Tensor<B, 4>,
        obj_mask: &Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let eps = 1e-7;

        // Apply sigmoid
        let p = sigmoid(pred.clone()).clamp(eps, 1.0 - eps);

        // Expand obj_mask for all classes
        let [_, num_classes, _, _] = pred.dims();
        let obj_mask = obj_mask.clone().repeat_dim(1, num_classes);

        // BCE formula
        let pos_loss = target.clone() * p.clone().log();
        let neg_loss = (Tensor::ones_like(target) - target.clone()) * 
                       (Tensor::ones_like(&p) - p).log();

        let loss = -(pos_loss + neg_loss) * obj_mask.clone();

        // Normalize
        let num_objects = obj_mask.clone().sum().clamp_min(1.0);
        loss.sum() / num_objects
    }
}