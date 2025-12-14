pub mod model;
pub mod training;
pub mod dataset;
pub mod gui;

// Re-export main items
pub use model::{YOLOv8Nano, Backbone, Neck, DetectionHead, YOLOLoss};
pub use training::{Trainer, TrainingConfig, TrainingState, TrainingMetrics};
pub use dataset::{RoadVehicleDataset, DatasetConfig, BatchLoader};
pub use gui::TrainingVisualizerApp;