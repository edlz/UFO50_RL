pub mod capture;
pub mod input;

use self::input::Input;
use super::GameRunner;
use std::sync::mpsc;

/// Win32 GameRunner: spawns a capture thread with a message pump,
/// communicates via channels. Training thread calls next_frame/execute_action.
pub struct Win32Runner {
    frame_rx: mpsc::Receiver<Vec<u8>>,
    action_tx: mpsc::SyncSender<usize>,
    reset_tx: mpsc::Sender<Vec<usize>>,
    width: u32,
    height: u32,
}

impl Win32Runner {
    /// Create runner + main loop closure. Runner goes to training thread,
    /// closure must be called on the main thread (Win32 message pump).
    pub fn new(
        window_title: &str,
        width: u32,
        height: u32,
    ) -> windows::core::Result<(Self, impl FnOnce() -> windows::core::Result<()>)> {
        capture::init()?;

        let mut input = Input::new(window_title)?;

        let (frame_tx, frame_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let (action_tx, action_rx) = mpsc::sync_channel::<usize>(1);
        let (reset_tx, reset_rx) = mpsc::channel::<Vec<usize>>();

        let title = window_title.to_string();
        let w = width;
        let h = height;

        // Send initial NOOP to unblock first frame read
        let action_tx_clone = action_tx.clone();
        let _ = action_tx_clone.send(0);

        let main_loop = move || {
            capture::run(
                &title,
                move |crop, frame, reader: &mut capture::FrameReader| {
                    // Check for reset signal (non-blocking)
                    if let Ok(extra_keys) = reset_rx.try_recv() {
                        if extra_keys.is_empty() {
                            input.release_all();
                        } else {
                            input.reset_game(&extra_keys);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }

                    // Wait for action from training thread
                    let action = match action_rx.recv() {
                        Ok(a) => a,
                        Err(_) => return false,
                    };

                    input.execute_action(action);

                    let pixels = match reader.read_cropped(frame, crop, w, h) {
                        Ok(p) => p.to_vec(),
                        Err(e) => {
                            eprintln!("FATAL read error: {}", e);
                            return false;
                        }
                    };

                    frame_tx.send(pixels).is_ok()
                },
            )
        };

        Ok((
            Self {
                frame_rx,
                action_tx,
                reset_tx,
                width,
                height,
            },
            main_loop,
        ))
    }
}

impl GameRunner for Win32Runner {
    fn next_frame(&mut self) -> Result<Vec<u8>, String> {
        self.frame_rx
            .recv()
            .map_err(|_| "Capture thread disconnected".to_string())
    }

    fn execute_action(&mut self, action: usize) {
        let _ = self.action_tx.send(action);
    }

    fn release_all(&mut self) {
        let _ = self.reset_tx.send(vec![]);
    }

    fn reset_game(&mut self, extra_keys: &[usize]) {
        let _ = self.reset_tx.send(extra_keys.to_vec());
    }

    fn obs_width(&self) -> u32 {
        self.width
    }

    fn obs_height(&self) -> u32 {
        self.height
    }
}
