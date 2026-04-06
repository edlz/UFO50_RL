use ufo50ppo::platform;
use ufo50ppo::util::{OBS_H, OBS_W, WINDOW_TITLE};

fn main() -> windows::core::Result<()> {
    platform::win32::capture::init()?;

    let mut input = platform::win32::input::Input::new(WINDOW_TITLE)?;

    let mut frame_count = 0u32;
    let mut snap_count = 0u32;
    let max_snaps = 10;
    let snap_interval = 180;

    std::fs::create_dir_all("debug_frames").ok();

    platform::win32::capture::run(
        WINDOW_TITLE,
        move |crop, frame, reader: &mut platform::win32::capture::FrameReader| {
            if let Err(e) = reader.read_cropped(frame, crop, OBS_W, OBS_H) {
                eprintln!("read error: {}", e);
                return true;
            }

            if frame_count % snap_interval == 0 {
                if snap_count >= max_snaps {
                    println!("Captured {} frames, done.", max_snaps);
                    input.release_all();
                    return false;
                }
                let path = format!("debug_frames/frame_{:03}.bmp", snap_count);
                match reader.save_debug_bmp(&path) {
                    Ok(()) => println!("Saved {} (frame {})", path, frame_count),
                    Err(e) => eprintln!("Save error: {}", e),
                }
                snap_count += 1;
            }

            frame_count += 1;
            true
        },
    )
}
