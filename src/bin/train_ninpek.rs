use std::sync::mpsc;
use ufo50ppo::games;
use ufo50ppo::games::GameTracker;
use ufo50ppo::platform;
use ufo50ppo::platform::GameRunner;
use ufo50ppo::train;
use ufo50ppo::util::{OBS_H, OBS_W, WINDOW_TITLE};

enum DebugMsg {
    Frame(Vec<u8>, &'static str),
    NewEpisode(u32),
}
const ROLLOUT_LEN: usize = 1024;
const LATEST_SAVE_INTERVAL: u64 = 10_000;
const VERSIONED_SAVE_INTERVAL: u64 = 250_000;
const LEARNING_RATE: f64 = 3e-4;
const GAMMA: f64 = 0.99;
const GAE_LAMBDA: f64 = 0.95;

const EPISODE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

use ufo50ppo::util::checkpoint::{self, CheckpointMeta};

fn save_checkpoint(
    dir: &str,
    name: &str,
    model: &train::model::ActorCritic,
    episode: u32,
    frames: u64,
    updates: u64,
    best: f64,
) {
    let path = format!("{}/{}.safetensors", dir, name);
    model
        .vs
        .save(&path)
        .unwrap_or_else(|e| eprintln!("Save error: {}", e));
    checkpoint::save_metadata(
        dir,
        name,
        &CheckpointMeta {
            game: "ninpek",
            resolution: (OBS_W, OBS_H),
            episode,
            total_frames: frames,
            ppo_updates: updates,
            best_reward: best,
            rollout_len: ROLLOUT_LEN,
            learning_rate: LEARNING_RATE,
            gamma: GAMMA,
            gae_lambda: GAE_LAMBDA,
        },
    );
}

fn explained_variance(values: &[f64], returns: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_r = returns.iter().sum::<f64>() / n;
    let var_r = returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n;
    if var_r < 1e-8 {
        return 0.0;
    }
    let var_diff = values
        .iter()
        .zip(returns)
        .map(|(v, r)| (r - v).powi(2))
        .sum::<f64>()
        / n;
    1.0 - var_diff / var_r
}

fn print_episode_breakdown(scores: u32, life_gained: u32, life_lost: u32, survival: f64) {
    println!(
        "  scores: {} | life+: {} | life-: {} | survival: {:.1}",
        scores, life_gained, life_lost, survival,
    );
}

fn drain_until_gameplay(
    runner: &mut dyn GameRunner,
    tracker: &dyn GameTracker,
    total_frames: &mut u64,
) -> bool {
    loop {
        match runner.next_frame() {
            Ok(pixels) => {
                runner.execute_action(0);
                *total_frames += 1;
                if !tracker.is_menu_screen(&pixels) {
                    return true;
                }
            }
            Err(e) => {
                eprintln!("Frame source closed during drain: {}", e);
                return false;
            }
        }
    }
}

struct TrainingConfig {
    max_episodes: Option<u32>,
    max_frames: Option<u64>,
    max_minutes: Option<u64>,
    auto_resume: bool,
    debug: bool,
    debug_tx: Option<mpsc::SyncSender<DebugMsg>>,
    checkpoint_dir: String,
    runs_dir: String,
}

fn training_thread(mut runner: Box<dyn GameRunner>, cfg: TrainingConfig) {
    let TrainingConfig {
        max_episodes,
        max_frames,
        max_minutes,
        auto_resume,
        debug,
        debug_tx,
        checkpoint_dir,
        runs_dir,
    } = cfg;
    let training_start = std::time::Instant::now();
    let max_duration = max_minutes.map(|m| std::time::Duration::from_secs(m * 60));
    let w = runner.obs_width();
    let h = runner.obs_height();
    let device = tch::Device::cuda_if_available();
    println!("Training on: {:?}", device);

    let mut model = train::model::ActorCritic::new(device);
    let mut opt = model.optimizer(LEARNING_RATE);
    let mut frame_stack = train::preprocess::FrameStack::new(device);
    let mut buffer = train::ppo::RolloutBuffer::new(ROLLOUT_LEN);
    let ppo_cfg = train::ppo::PpoConfig {
        minibatch_size: 128,
        ..train::ppo::PpoConfig::default()
    };
    // Training thread passes absolute total_frames as step, no offset needed
    let mut logger = ufo50ppo::util::logger::TbLogger::new(&runs_dir, 0);
    let mut tracker = games::ninpek::NinpekTracker::new(w);

    let mut episode = 0u32;
    let mut episode_reward = 0.0f64;
    let mut ep_scores = 0u32;
    let mut ep_life_gained = 0u32;
    let mut ep_life_lost = 0u32;
    let mut ep_survival = 0.0f64;
    let mut episode_frames = 0u32;
    let mut total_frames = 0u64;
    let mut update_count = 0u64;
    let mut best_reward = f64::NEG_INFINITY;

    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");

    // Resume from latest checkpoint if it exists
    match checkpoint::try_load(&checkpoint_dir, "latest", &mut model.vs) {
        Ok(Some(state)) => {
            episode = state.episode;
            total_frames = state.total_frames;
            update_count = state.ppo_updates;
            best_reward = state.best_reward;
            println!(
                "Resuming: ep={} frames={} updates={} best={:.1}",
                episode, total_frames, update_count, best_reward
            );
        }
        Ok(None) => {}
        Err(e) => eprintln!("Failed to load latest checkpoint: {}", e),
    }

    // Send initial NOOP to unblock first capture frame
    runner.execute_action(0);

    let mut t_recv = std::time::Duration::ZERO;
    let mut t_preprocess = std::time::Duration::ZERO;
    let mut t_act = std::time::Duration::ZERO;
    let mut t_tracker = std::time::Duration::ZERO;
    let mut t_ppo = std::time::Duration::ZERO;
    let mut timing_frames = 0u32;
    let mut last_timing_print = std::time::Instant::now();
    let mut episode_start = std::time::Instant::now();

    macro_rules! reset_episode_state {
        () => {{
            runner.reset_game(tracker.extra_reset_keys());
            if !drain_until_gameplay(&mut *runner, &tracker, &mut total_frames) {
                return;
            }
            frame_stack.reset();
            tracker = games::ninpek::NinpekTracker::new(w);
            episode += 1;
            if let Some(ref dtx) = debug_tx {
                let _ = dtx.try_send(DebugMsg::NewEpisode(episode));
            }
            episode_reward = 0.0;
            episode_frames = 0;
            ep_scores = 0;
            ep_life_gained = 0;
            ep_life_lost = 0;
            ep_survival = 0.0;
            episode_start = std::time::Instant::now();
        }};
    }

    // Reset game before starting
    runner.reset_game(tracker.extra_reset_keys());
    if !drain_until_gameplay(&mut *runner, &tracker, &mut total_frames) {
        return;
    }

    loop {
        let t0 = std::time::Instant::now();
        let pixels = match runner.next_frame() {
            Ok(p) => p,
            Err(_) => break,
        };
        t_recv += t0.elapsed();

        // Menu frames: skip model/buffer, but still check for game over via tracker
        // (leaderboard detection needs 2-frame state in process_frame)
        if tracker.is_menu_screen(&pixels) {
            let result = tracker.process_frame(&pixels); // safe: leaderboard check returns early before life/score
            runner.execute_action(0);
            total_frames += 1;
            if result.done {
                // Game over detected on a menu frame — trigger episode reset
                let reason = if result.event_name.is_empty() {
                    "DONE"
                } else {
                    result.event_name
                };
                println!(
                    "\rEpisode {:4} | {} | reward: {:+.1} | frames: {} | total: {} | updates: {}          ",
                    episode, reason, episode_reward, episode_frames, total_frames, update_count
                );
                if debug {
                    print_episode_breakdown(ep_scores, ep_life_gained, ep_life_lost, ep_survival);
                }
                logger.log_episode(total_frames as usize, episode_reward, episode_frames);
                if episode_reward > best_reward {
                    best_reward = episode_reward;
                    save_checkpoint(
                        &checkpoint_dir,
                        "best",
                        &model,
                        episode,
                        total_frames,
                        update_count,
                        best_reward,
                    );
                    println!("  New best: {:+.1}", best_reward);
                }
                reset_episode_state!();
                // Run PPO on partial rollout before clearing
                if buffer.len() >= 32 {
                    let last_value = 0.0; // episode ended
                    let (advantages, returns) = train::ppo::compute_gae(
                        &buffer.rewards,
                        &buffer.values,
                        &buffer.dones,
                        last_value,
                        GAMMA,
                        GAE_LAMBDA,
                    );
                    let stats = train::ppo::update(
                        &mut model,
                        &mut opt,
                        &buffer,
                        &advantages,
                        &returns,
                        &ppo_cfg,
                    );
                    update_count += 1;
                    logger.log_update(
                        total_frames as usize,
                        stats.policy_loss,
                        stats.value_loss,
                        stats.entropy,
                        stats.total_loss,
                    );
                    logger.log_learning_rate(total_frames as usize, LEARNING_RATE);
                    let ev = explained_variance(&buffer.values, &returns);
                    logger.log_explained_variance(total_frames as usize, ev);
                }
                buffer.clear();
                if let Some(max) = max_episodes {
                    if episode >= max {
                        println!("Reached {} episodes, stopping.", max);
                        break;
                    }
                }
            }
            continue;
        }

        let t1 = std::time::Instant::now();
        let obs = frame_stack.push(&pixels, w, h);
        t_preprocess += t1.elapsed();

        let t2 = std::time::Instant::now();
        let (action, log_prob, value) = model.act(&obs);
        t_act += t2.elapsed();

        runner.execute_action(action as usize);

        let t3 = std::time::Instant::now();
        let result = tracker.process_frame(&pixels);
        t_tracker += t3.elapsed();
        let reward = result.reward;
        let done = result.done;

        episode_reward += reward;
        episode_frames += 1;
        total_frames += 1;

        match result.event_name {
            "SCORE" => ep_scores += 1,
            "LIFE+" => ep_life_gained += 1,
            "LIFE-" => ep_life_lost += 1,
            "" => ep_survival += reward,
            _ => {}
        }

        // Check time limit every 100 frames to avoid per-frame syscall
        let time_up =
            max_duration.is_some_and(|d| total_frames % 100 == 0 && training_start.elapsed() >= d);
        if time_up || max_frames.is_some_and(|max| total_frames >= max) {
            if time_up {
                println!("\rReached {} minutes, stopping.", max_minutes.unwrap());
            } else {
                println!("\rReached {} frames, stopping.", max_frames.unwrap());
            }
            // Flush partial rollout
            if buffer.len() >= 32 {
                let last_value = value;
                let (advantages, returns) = train::ppo::compute_gae(
                    &buffer.rewards,
                    &buffer.values,
                    &buffer.dones,
                    last_value,
                    GAMMA,
                    GAE_LAMBDA,
                );
                let stats = train::ppo::update(
                    &mut model,
                    &mut opt,
                    &buffer,
                    &advantages,
                    &returns,
                    &ppo_cfg,
                );
                update_count += 1;
                logger.log_update(
                    total_frames as usize,
                    stats.policy_loss,
                    stats.value_loss,
                    stats.entropy,
                    stats.total_loss,
                );
                logger.log_explained_variance(
                    total_frames as usize,
                    explained_variance(&buffer.values, &returns),
                );
            }
            save_checkpoint(
                &checkpoint_dir,
                "latest",
                &model,
                episode,
                total_frames,
                update_count,
                best_reward,
            );
            break;
        }

        // Episode timeout — reload latest checkpoint and resume (or exit)
        if episode_start.elapsed() > EPISODE_TIMEOUT {
            eprintln!(
                "\rEpisode {:4} TIMEOUT after {:?}",
                episode,
                episode_start.elapsed()
            );
            if !auto_resume {
                eprintln!("Exiting due to timeout.");
                return;
            }
            match checkpoint::try_load(&checkpoint_dir, "latest", &mut model.vs) {
                Ok(Some(state)) => {
                    episode = state.episode;
                    total_frames = state.total_frames;
                    update_count = state.ppo_updates;
                    best_reward = state.best_reward;
                    eprintln!(
                        "  Reloaded latest: ep={} frames={} updates={}",
                        episode, total_frames, update_count
                    );
                }
                Ok(None) => {
                    eprintln!("  No latest checkpoint to reload, exiting.");
                    return;
                }
                Err(e) => {
                    eprintln!("  Failed to reload latest: {}", e);
                    return;
                }
            }
            buffer.clear();
            reset_episode_state!();
            continue;
        }

        // Store in rollout buffer
        buffer.push(obs, action, log_prob, reward, value, done);

        timing_frames += 1;

        // PPO update when buffer full OR episode ended with enough data
        if buffer.len() >= ROLLOUT_LEN || (done && buffer.len() >= 32) {
            let t4 = std::time::Instant::now();
            // Bootstrap: V(s_{T+1}). If episode ended, future value is 0.
            // Otherwise use value of the last state (close approximation at 47fps).
            let last_value = if done { 0.0 } else { value };
            let (advantages, returns) = train::ppo::compute_gae(
                &buffer.rewards,
                &buffer.values,
                &buffer.dones,
                last_value,
                GAMMA,
                GAE_LAMBDA,
            );
            let stats = train::ppo::update(
                &mut model,
                &mut opt,
                &buffer,
                &advantages,
                &returns,
                &ppo_cfg,
            );
            t_ppo += t4.elapsed();
            let ev = explained_variance(&buffer.values, &returns);
            buffer.clear();
            update_count += 1;

            logger.log_update(
                total_frames as usize,
                stats.policy_loss,
                stats.value_loss,
                stats.entropy,
                stats.total_loss,
            );
            logger.log_learning_rate(total_frames as usize, LEARNING_RATE);
            logger.log_explained_variance(total_frames as usize, ev);
        }

        if let Some(ref dtx) = debug_tx {
            let _ = dtx.try_send(DebugMsg::Frame(pixels, result.event_name));
        }

        // Print timing metrics once ever 5 minutes
        if timing_frames > 0 && last_timing_print.elapsed() >= std::time::Duration::from_secs(300) {
            let n = timing_frames as f64;
            println!(
                "\r[timing] recv: {:.1}ms  preprocess: {:.1}ms  act: {:.1}ms  tracker: {:.1}ms  ppo: {:.1}ms (per frame avg)     ",
                t_recv.as_secs_f64() / n * 1000.0,
                t_preprocess.as_secs_f64() / n * 1000.0,
                t_act.as_secs_f64() / n * 1000.0,
                t_tracker.as_secs_f64() / n * 1000.0,
                t_ppo.as_secs_f64() / n * 1000.0,
            );
            let fps = n / (t_recv + t_preprocess + t_act + t_tracker + t_ppo).as_secs_f64();
            logger.log_fps(total_frames as usize, fps);
            t_recv = std::time::Duration::ZERO;
            t_preprocess = std::time::Duration::ZERO;
            t_act = std::time::Duration::ZERO;
            t_tracker = std::time::Duration::ZERO;
            t_ppo = std::time::Duration::ZERO;
            timing_frames = 0;
            last_timing_print = std::time::Instant::now();
        }

        // Live status line
        if episode_frames % 10 == 0 {
            print!(
                "\rEp {:3} | reward: {:+7.1} | lives: {} | frame: {} | updates: {} | {}     ",
                episode,
                episode_reward,
                result.lives,
                episode_frames,
                update_count,
                result.event_name
            );
        }

        // Episode end
        if done {
            let reason = if result.event_name.is_empty() {
                "DONE"
            } else {
                result.event_name
            };
            println!(
                "\rEpisode {:4} | {} | reward: {:+.1} | frames: {} | total: {} | updates: {}          ",
                episode, reason, episode_reward, episode_frames, total_frames, update_count
            );
            if debug {
                print_episode_breakdown(ep_scores, ep_life_gained, ep_life_lost, ep_survival);
            }

            logger.log_episode(total_frames as usize, episode_reward, episode_frames);

            // Save best model
            if episode_reward > best_reward {
                best_reward = episode_reward;
                save_checkpoint(
                    &checkpoint_dir,
                    "best",
                    &model,
                    episode,
                    total_frames,
                    update_count,
                    best_reward,
                );
                println!("  New best: {:+.1}", best_reward);
            }

            reset_episode_state!();

            if let Some(max) = max_episodes {
                if episode >= max {
                    println!("Reached {} episodes, stopping.", max);
                    save_checkpoint(
                        &checkpoint_dir,
                        "latest",
                        &model,
                        episode,
                        total_frames,
                        update_count,
                        best_reward,
                    );
                    break;
                }
            }
        }

        // Periodic save: latest (frequent, for recovery) and versioned (sparse, for archival)
        if total_frames > 0 && total_frames % LATEST_SAVE_INTERVAL == 0 {
            save_checkpoint(
                &checkpoint_dir,
                "latest",
                &model,
                episode,
                total_frames,
                update_count,
                best_reward,
            );
        }
        if total_frames > 0 && total_frames % VERSIONED_SAVE_INTERVAL == 0 {
            save_checkpoint(
                &checkpoint_dir,
                &format!("frame_{:08}", total_frames),
                &model,
                episode,
                total_frames,
                update_count,
                best_reward,
            );
        }
    }
}

fn spawn_debug_thread() -> mpsc::SyncSender<DebugMsg> {
    let (tx, rx) = mpsc::sync_channel::<DebugMsg>(1);
    std::thread::spawn(move || {
        let mut ep = 0u32;
        let mut frame = 0u32;
        let w = OBS_W;
        let h = OBS_H;
        let row_bytes = (w * 3 + 3) & !3;
        let pixel_size = row_bytes * h;
        let file_size = 54 + pixel_size;
        let mut row_buf = vec![0u8; row_bytes as usize];

        while let Ok(msg) = rx.recv() {
            match msg {
                DebugMsg::NewEpisode(n) => {
                    ep = n;
                    frame = 0;
                    std::fs::create_dir_all(format!("debug_frames/ep_{:04}", ep)).ok();
                }
                DebugMsg::Frame(pixels, event) => {
                    let path = if event == "SCORE" {
                        format!(
                            "debug_frames/ep_{:04}/{:05}_+{}.bmp",
                            ep,
                            frame,
                            games::ninpek::rewards::SCORE_UP as i64
                        )
                    } else {
                        format!("debug_frames/ep_{:04}/{:05}.bmp", ep, frame)
                    };
                    if let Ok(mut f) = std::fs::File::create(&path) {
                        use std::io::Write;
                        let _ = f.write_all(b"BM");
                        let _ = f.write_all(&file_size.to_le_bytes());
                        let _ = f.write_all(&0u32.to_le_bytes());
                        let _ = f.write_all(&54u32.to_le_bytes());
                        let _ = f.write_all(&40u32.to_le_bytes());
                        let _ = f.write_all(&(w as i32).to_le_bytes());
                        let _ = f.write_all(&(-(h as i32)).to_le_bytes());
                        let _ = f.write_all(&1u16.to_le_bytes());
                        let _ = f.write_all(&24u16.to_le_bytes());
                        let _ = f.write_all(&0u32.to_le_bytes());
                        let _ = f.write_all(&pixel_size.to_le_bytes());
                        let _ = f.write_all(&[0u8; 16]);
                        for y in 0..h as usize {
                            for x in 0..w as usize {
                                let src = y * w as usize * 4 + x * 4;
                                let dst = x * 3;
                                row_buf[dst] = pixels[src];
                                row_buf[dst + 1] = pixels[src + 1];
                                row_buf[dst + 2] = pixels[src + 2];
                            }
                            let _ = f.write_all(&row_buf);
                        }
                    }
                    frame += 1;
                }
            }
        }
    });
    let _ = tx.send(DebugMsg::NewEpisode(0));
    tx
}

fn main() -> windows::core::Result<()> {
    let args = ufo50ppo::util::cli::parse_train_args("default");
    let checkpoint_dir = format!("checkpoints/ninpek/{}", args.namespace);
    let runs_dir = format!("runs/ninpek/{}", args.namespace);

    // Pre-load torch_cuda.dll
    unsafe {
        windows::Win32::System::LibraryLoader::LoadLibraryA(windows::core::s!("torch_cuda.dll"))
            .ok();
    }

    let (runner, main_loop) = platform::win32::Win32Runner::new(WINDOW_TITLE, OBS_W, OBS_H)?;

    let debug_tx = if args.debug {
        Some(spawn_debug_thread())
    } else {
        None
    };

    // Spawn training thread — uses GameRunner trait, platform-agnostic
    std::thread::spawn(move || {
        training_thread(
            Box::new(runner),
            TrainingConfig {
                max_episodes: args.max_episodes,
                max_frames: args.max_frames,
                max_minutes: args.max_minutes,
                auto_resume: args.auto_resume,
                debug: args.debug,
                debug_tx,
                checkpoint_dir,
                runs_dir,
            },
        );
    });

    println!(
        "Training Ninpek | ns: {} | {}x{}",
        args.namespace, OBS_W, OBS_H
    );
    if let Some(n) = args.max_episodes {
        println!("  Max episodes: {}", n);
    }
    if args.debug {
        println!("Debug mode: saving frames to debug_frames/");
    }
    println!();

    // Win32 message pump must run on main thread
    main_loop()
}
