//! Reinforcement learning for adaptive source selection
//!
//! This module implements online learning and feedback mechanisms
//! to continuously improve source selection based on query outcomes.

mod feedback;
mod policy;
mod reward;

pub use feedback::Feedback;
pub use policy::{Policy, PolicyType};
pub use reward::Reward;
