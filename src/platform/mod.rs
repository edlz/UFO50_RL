/// Platform implementations live in submodules (e.g., win32/, linux/).
/// Each must implement GameRunner: frame capture, input execution, and game lifecycle.
pub mod win32;

/// Platform-agnostic interface for running a game.
/// Combines frame capture, input, and game lifecycle.
/// On Windows: wraps capture::run callback + Input via channels.
/// On Linux: would be a single-threaded X11/PipeWire + uinput loop.
pub trait GameRunner: Send {
    /// Block until next frame is available. Returns BGRA pixels at the configured resolution.
    fn next_frame(&mut self) -> Result<Vec<u8>, String>;

    /// Execute a discrete action by index.
    fn execute_action(&mut self, action: usize);

    /// Release all held keys/buttons.
    fn release_all(&mut self);

    /// Reset the game to initial state with optional extra key presses.
    fn reset_game(&mut self, extra_keys: &[usize]);

    /// Observation width.
    fn obs_width(&self) -> u32;

    /// Observation height.
    fn obs_height(&self) -> u32;
}

/// Number of discrete actions available.
pub const NUM_ACTIONS: usize = 26;

/// Human-readable action names.
pub const ACTION_NAMES: &[&str] = &[
    "NOOP",
    "Up",
    "Down",
    "Left",
    "Right",
    "Up-Right",
    "Up-Left",
    "Down-Right",
    "Down-Left",
    "A",
    "B",
    "Up+A",
    "Up+B",
    "Down+A",
    "Left+A",
    "Right+A",
    "Left+B",
    "Right+B",
    "Up-Right+A",
    "Up-Right+B",
    "Up-Left+A",
    "Up-Left+B",
    "Down-Right+A",
    "Down-Right+B",
    "Down-Left+A",
    "Down-Left+B",
];
