#[derive(Debug, Clone)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub confidence: f32,
    pub class_id: usize,
}

#[derive(Debug, Clone)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: usize,
}

impl Detection {
    pub fn to_bbox(&self) -> BBox {
        BBox {
            x1: self.x - self.w / 2.0,
            y1: self.y - self.h / 2.0,
            x2: self.x + self.w / 2.0,
            y2: self.y + self.h / 2.0,
            confidence: self.confidence,
            class_id: self.class_id,
        }
    }
}

pub struct NMS;

impl NMS {
    pub fn iou(box1: &BBox, box2: &BBox) -> f32 {
        let x1_inter = box1.x1.max(box2.x1);
        let y1_inter = box1.y1.max(box2.y1);
        let x2_inter = box1.x2.min(box2.x2);
        let y2_inter = box1.y2.min(box2.y2);

        if x2_inter < x1_inter || y2_inter < y1_inter {
            return 0.0;
        }

        let intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter);
        let box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
        let box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
        let union = box1_area + box2_area - intersection;

        intersection / union.max(1e-6)
    }

    pub fn apply(
        detections: Vec<Detection>,
        iou_threshold: f32,
        confidence_threshold: f32,
    ) -> Vec<Detection> {
        let mut result = Vec::new();

        let mut valid: Vec<_> = detections
            .into_iter()
            .filter(|d| d.confidence >= confidence_threshold)
            .collect();

        if valid.is_empty() {
            return result;
        }

        valid.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut used = vec![false; valid.len()];

        for i in 0..valid.len() {
            if used[i] {
                continue;
            }

            let current = valid[i].clone();
            result.push(current.clone());

            let current_box = current.to_bbox();

            for j in (i + 1)..valid.len() {
                if used[j] || valid[j].class_id != current.class_id {
                    continue;
                }

                let other_box = valid[j].to_bbox();
                let iou = Self::iou(&current_box, &other_box);

                if iou > iou_threshold {
                    used[j] = true;
                }
            }
        }

        result.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });

        result
    }
}