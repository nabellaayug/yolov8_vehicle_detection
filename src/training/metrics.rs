use crate::model::Detection;

pub struct Metrics {
    pub map_50: f32,
    pub map_50_95: f32,
    pub precision: f32,
    pub recall: f32,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            map_50: 0.0,
            map_50_95: 0.0,
            precision: 0.0,
            recall: 0.0,
        }
    }
    
    pub fn calculate_map(
        predictions: &[Vec<Detection>],
        ground_truths: &[Vec<Detection>],
        num_classes: usize,
        iou_threshold: f32,
    ) -> f32 {
        // TODO: Implement proper mAP calculation
        // This is a placeholder
        let mut ap_sum = 0.0;
        
        for class_id in 0..num_classes {
            let ap = Self::calculate_ap_for_class(
                predictions,
                ground_truths,
                class_id,
                iou_threshold,
            );
            ap_sum += ap;
        }
        
        ap_sum / num_classes as f32
    }
    
    fn calculate_ap_for_class(
        _predictions: &[Vec<Detection>],
        _ground_truths: &[Vec<Detection>],
        _class_id: usize,
        _iou_threshold: f32,
    ) -> f32 {
        // TODO: Implement AP calculation per class
        // 1. Match predictions to ground truths
        // 2. Calculate precision-recall curve
        // 3. Calculate area under curve
        0.0
    }
    
    pub fn iou(det1: &Detection, det2: &Detection) -> f32 {
        let x1_min = det1.x - det1.w / 2.0;
        let y1_min = det1.y - det1.h / 2.0;
        let x1_max = det1.x + det1.w / 2.0;
        let y1_max = det1.y + det1.h / 2.0;
        
        let x2_min = det2.x - det2.w / 2.0;
        let y2_min = det2.y - det2.h / 2.0;
        let x2_max = det2.x + det2.w / 2.0;
        let y2_max = det2.y + det2.h / 2.0;
        
        let x_inter = (x1_max.min(x2_max) - x1_min.max(x2_min)).max(0.0);
        let y_inter = (y1_max.min(y2_max) - y1_min.max(y2_min)).max(0.0);
        
        let intersection = x_inter * y_inter;
        let area1 = det1.w * det1.h;
        let area2 = det2.w * det2.h;
        let union = area1 + area2 - intersection;
        
        intersection / union.max(1e-6)
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}