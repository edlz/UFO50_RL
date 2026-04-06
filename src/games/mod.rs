pub mod ninpek;

/// Pixel region for game-specific screen area detection.
/// Coordinates are calibrated for the game's capture resolution.
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}
