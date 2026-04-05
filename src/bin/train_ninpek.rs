use ufo50ppo::game;

const OBS_W: u32 = 128;
const OBS_H: u32 = 128;
const WINDOW_TITLE: &str = "UFO 50";
const EXTRA_RESET_KEYS: &[usize] = &[game::input::VK_Z];

fn main() -> windows::core::Result<()> {
    // Pre-load torch_cuda.dll so PyTorch's lazy CUDA init can find it.
    unsafe {
        windows::Win32::System::LibraryLoader::LoadLibraryA(windows::core::s!("torch_cuda.dll"))
            .ok();
    }

    game::capture::init()?;

    let mut input = game::input::Input::new(WINDOW_TITLE)?;
    let mut tracker = game::games::ninpek::NinpekTracker::new(OBS_W);

    let mut episode = 0u32;
    let mut episode_reward = 0.0f64;
    let mut frame_count = 0u32;
    let mut total_frames = 0u64;
    let mut last_fps_time = std::time::Instant::now();
    let mut fps_frames = 0u32;
    let mut fps = 0.0f64;
    let mut last_event = String::new();

    println!("Training Ninpek | Capture: {}x{}", OBS_W, OBS_H);
    println!("Device: {:?}", tch::Device::cuda_if_available());
    println!();

    game::capture::run(
        WINDOW_TITLE,
        move |crop, frame, reader: &mut game::capture::FrameReader| {
            let pixels = match reader.read_cropped(frame, crop, OBS_W, OBS_H) {
                Ok(p) => p.to_vec(),
                Err(e) => {
                    eprintln!("read error: {}", e);
                    return true;
                }
            };

            let result = tracker.process_frame(&pixels);
            episode_reward += result.reward;
            frame_count += 1;
            total_frames += 1;
            fps_frames += 1;

            // FPS tracking
            let elapsed = last_fps_time.elapsed();
            if elapsed.as_secs() >= 1 {
                fps = fps_frames as f64 / elapsed.as_secs_f64();
                fps_frames = 0;
                last_fps_time = std::time::Instant::now();
            }

            // Track reward events
            use game::games::ninpek::RewardEvent;
            match result.event {
                RewardEvent::Survival => {}
                RewardEvent::ScoreUp => last_event = format!("SCORE +{:.0}", result.reward),
                RewardEvent::LifeGained => last_event = format!("LIFE+ +{:.0}", result.reward),
                RewardEvent::LifeLost => last_event = format!("LIFE- {:.0}", result.reward),
                RewardEvent::StageComplete => last_event = format!("STAGE +{:.0}", result.reward),
                RewardEvent::GameComplete | RewardEvent::GameOver => {}
            }

            // Live status line (overwrite in place)
            if frame_count % 10 == 0 {
                print!(
                    "\rEp {:3} | reward: {:+7.1} | lives: {} | fps: {:.0} | frame: {} | {}     ",
                    episode, episode_reward, result.lives, fps, frame_count, last_event
                );
            }

            // TODO: feed pixels to model, get action, execute

            if result.done {
                let reason = match result.event {
                    RewardEvent::GameOver => "GAME OVER",
                    RewardEvent::GameComplete => "WIN",
                    _ => "DONE",
                };
                println!(
                    "\rEpisode {:4} | {} | reward: {:+.1} | frames: {} | total: {}          ",
                    episode, reason, episode_reward, frame_count, total_frames
                );

                input.reset_game(EXTRA_RESET_KEYS);
                std::thread::sleep(std::time::Duration::from_millis(1000));

                tracker = game::games::ninpek::NinpekTracker::new(OBS_W);
                episode += 1;
                episode_reward = 0.0;
                frame_count = 0;
                last_event.clear();
            }

            true
        },
    )
}
