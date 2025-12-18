pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    pub best_loss: f32,
    counter: usize,
    pub stopped: bool,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f32::INFINITY,
            counter: 0,
            stopped: false,
        }
    }
    
    pub fn should_stop(&mut self, current_loss: f32) -> bool {
        if self.stopped {
            return true;
        }
        
        // Check if loss improved significantly
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.counter = 0;
            false
        } else {
            self.counter += 1;
            
            if self.counter >= self.patience {
                self.stopped = true;
                println!("Early stopping triggered! No improvement for {} epochs", self.patience);
                true
            } else {
                false
            }
        }
    }
    
    pub fn reset(&mut self) {
        self.best_loss = f32::INFINITY;
        self.counter = 0;
        self.stopped = false;
    }
}