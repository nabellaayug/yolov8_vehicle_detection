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
        log::info!("YOLOLoss compute:");
        log::info!("  pred_p2: {:?}, target_p2: {:?}", pred_p2.dims(), target_p2.dims());
        log::info!("  pred_p3: {:?}, target_p3: {:?}", pred_p3.dims(), target_p3.dims());
        log::info!("  pred_p4: {:?}, target_p4: {:?}", pred_p4.dims(), target_p4.dims());
        
        let loss_p2 = Self::scale_loss(pred_p2, target_p2);
        let loss_p3 = Self::scale_loss(pred_p3, target_p3);
        let loss_p4 = Self::scale_loss(pred_p4, target_p4);
        
        // Combine losses - simple mean
        let total_loss = (loss_p2 + loss_p3 + loss_p4) / 3.0;
        
        log::info!("  total_loss: {:?}", total_loss.dims());
        
        total_loss.reshape([1])
    }

    fn scale_loss<B: Backend>(
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let pred_dims = predictions.dims();
        let target_dims = targets.dims();
        
        log::debug!("  scale_loss: pred={:?}, target={:?}", pred_dims, target_dims);
        
        // ✅ Simple approach: just compute MSE directly
        // Both should have same shape now
        if pred_dims != target_dims {
            log::warn!("    Shape mismatch! Padding/slicing to match...");
            // If shapes don't match, take minimum and slice
            let [b, c, h, w] = pred_dims;
            let [b_t, c_t, h_t, w_t] = target_dims;
            
            let min_c = c.min(c_t);
            let min_h = h.min(h_t);
            let min_w = w.min(w_t);
            
            let pred_slice = predictions.slice([0..b, 0..min_c, 0..min_h, 0..min_w]);
            let target_slice = targets.slice([0..b_t, 0..min_c, 0..min_h, 0..min_w]);
            
            let diff = pred_slice - target_slice;
            return (diff.clone() * diff).mean().reshape([1]);
        }
        
        // ✅ MSE loss - both shapes match
        let diff = predictions - targets;
        let mse = (diff.clone() * diff).mean();
        
        log::debug!("    MSE: {}", mse.to_data());
        
        mse.reshape([1])
    }
}