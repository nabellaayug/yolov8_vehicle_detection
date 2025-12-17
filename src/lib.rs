pub mod model;
pub mod data;
pub mod training;
pub mod gui;

// Re-exports for convenience
pub use model::{YOLOv8, Backbone, Neck, DetectionHead, Yolov8Loss, Detection, NMS};
pub use data::{YoloDataset, YoloDataLoader, BoundingBox};
pub use training::{Trainer, TrainingConfig, EarlyStopping};
pub use gui::TrainingGui;