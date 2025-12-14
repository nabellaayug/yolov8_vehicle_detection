use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

#[derive(Debug, Module)]
pub struct Concat;

impl Concat {
    pub fn new() -> Self { Self }
}

impl Concat {
    pub fn concat<B: Backend>(&self, tensors: &[Tensor<B, 4>]) -> Tensor<B, 4> {
        // Concatenate on channel dim
        // Depending on burn version, adjust to correct API
        Tensor::concat(tensors.to_vec(), 1)
    }
}
