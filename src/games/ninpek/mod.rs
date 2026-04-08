// game_over pixel regions are calibrated for 128x128 capture resolution. Score and
// lives are read from process memory (see mem.rs).
pub mod events;
mod game_over;
mod mem;
pub mod rewards;
mod tracker;

pub use game_over::*;
pub use tracker::NinpekTracker;

use crate::train::runner::{GameDefinition, Hyperparams};

pub const WINDOW_TITLE: &str = "UFO 50";

fn ninpek_debug_suffix(event: &str, reward: f64) -> String {
    if event == events::SCORE {
        format!("_+{}", reward as i64)
    } else {
        String::new()
    }
}

fn make_ninpek_tracker(w: u32) -> Box<dyn crate::games::GameTracker> {
    Box::new(NinpekTracker::new(w))
}

pub fn definition() -> GameDefinition {
    GameDefinition {
        name: "ninpek",
        window_title: WINDOW_TITLE,
        obs_width: 128,
        obs_height: 128,
        num_actions: crate::platform::NUM_ACTIONS,
        make_tracker: make_ninpek_tracker,
        hyperparams: Hyperparams::default(),
        debug_frame_suffix: ninpek_debug_suffix,
    }
}
