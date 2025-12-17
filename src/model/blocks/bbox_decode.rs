use burn::prelude::*;
use burn::tensor::activation::softmax;

#[derive(Debug, Clone)]
pub struct BBoxDecoder {
    pub reg_max: usize,
}

impl BBoxDecoder {
    pub fn new(reg_max: usize) -> Self {
        Self { reg_max }
    }

    pub fn decode<B: Backend>(&self, pred_reg: Tensor<B, 4>, stride: usize) -> Tensor<B, 3> {
        let bins = self.reg_max + 1;
        let [b, _, h, w] = pred_reg.dims();

        // Reshape to [B, 4, bins, H, W]
        let pred = pred_reg.reshape([b, 4, bins, h, w]);
        let prob = softmax(pred, 2);

        // Create projection tensor
        let proj_vec: Vec<f32> = (0..bins).map(|i| i as f32).collect();
        let proj: Tensor<B, 5> = Tensor::<B, 4>::from_floats(proj_vec.as_slice(), &prob.device())
            .reshape([1, 1, bins, 1, 1]);

        // Get distance predictions
        let dist = (prob * proj).sum_dim(2); // [B, 4, H, W]

        let device = dist.device();

        // Create grid - FIXED: Proper 2D grid generation
        let mut grid_y = Vec::new();
        for y in 0..h {
            for _x in 0..w {
                grid_y.push(y as f32);
            }
        }

        let mut grid_x = Vec::new();
        for _y in 0..h {
            for x in 0..w {
                grid_x.push(x as f32);
            }
        }

        // FIXED: Explicit type annotations
        let grid_x: Tensor<B, 3> = Tensor::<B, 4>::from_floats(grid_x.as_slice(), &device)
            .reshape([1, h, w])
            .repeat_dim(0, b);

        let grid_y: Tensor<B, 3> = Tensor::<B, 4>::from_floats(grid_y.as_slice(), &device)
            .reshape([1, h, w])
            .repeat_dim(0, b);

        // Calculate cell centers
        let cx = (grid_x.clone() + 0.5) * (stride as f32);
        let cy = (grid_y.clone() + 0.5) * (stride as f32);

        // Extract distances (left, top, right, bottom)
        let l = dist.clone().slice([0..b, 0..1, 0..h, 0..w]).squeeze::<3>() * (stride as f32);
        let t = dist.clone().slice([0..b, 1..2, 0..h, 0..w]).squeeze::<3>() * (stride as f32);
        let r = dist.clone().slice([0..b, 2..3, 0..h, 0..w]).squeeze::<3>() * (stride as f32);
        let b_dist = dist.slice([0..b, 3..4, 0..h, 0..w]).squeeze::<3>() * (stride as f32);

        // Convert to x1, y1, x2, y2
        let x1 = cx.clone() - l;
        let y1 = cy.clone() - t;
        let x2 = cx + r;
        let y2 = cy + b_dist;

        // Stack and reshape to [B, HW, 4]
        // FIXED: Explicit type annotation for stack result
        let stacked: Tensor<B, 4> = Tensor::stack(vec![x1, y1, x2, y2], 1); // [B, 4, H, W]
        stacked.reshape([b, 4, h * w]).swap_dims(1, 2) // [B, HW, 4]
    }
}