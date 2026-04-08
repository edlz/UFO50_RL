# ufo50ppo

PPO reinforcement learning agent that learns to play [UFO 50](https://50games.fun/) games by capturing the screen and sending keyboard inputs. Windows-only (Linux-portable architecture).

Currently trains on **Ninpek** (game 3). Score and lives are read directly from `ufo50.exe` process memory via pointer chains; stage completion and game over are detected from the captured frame pixels.

## Requirements

- **Windows 10/11**
- **Rust** (edition 2024)
- **libtorch** — set `LIBTORCH` env var to your libtorch path, add `lib/` to `PATH`
  - Set `LIBTORCH_BYPASS_VERSION_CHECK=1` if version mismatch
- **UFO 50** running with window title "UFO 50"

## Quick Start

```bash
# Train Ninpek
cargo run --release --bin train_ninpek

# Train with namespace (separate experiment)
cargo run --release --bin train_ninpek -- -n experiment1

# Train with limits (whichever comes first)
cargo run --release --bin train_ninpek -- -e 100 -f 500000
cargo run --release --bin train_ninpek -- -m 60

# Debug mode (saves frames + reward breakdown)
cargo run --release --bin train_ninpek -- -d

# View training progress
tensorboard --logdir runs/ninpek
```

## CLI Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--namespace` | `-n` | Training namespace (default: "default") |
| `--episodes` | `-e` | Max episodes before stopping |
| `--frames` | `-f` | Max frames before stopping |
| `--minutes` | `-m` | Max training time in minutes |
| `--auto-resume` | `-r` | On episode timeout, reload `latest` and continue instead of exiting |
| `--debug` | `-d` | Save frames to `debug_frames/ep_NNNN/`, print per-episode reward breakdown |

## Training Output

```
checkpoints/ninpek/{namespace}/
  latest.safetensors          # written every 10k frames + on clean exit
  best.safetensors            # highest episode reward
  frame_00250000.safetensors  # versioned archive every 250k frames

runs/ninpek/{namespace}/
  20260405_143022/            # tensorboard logs (timestamped per run)
```

Resuming automatically loads `latest.safetensors` and continues from the saved episode/frame/update count. With `--auto-resume`, an episode timeout (60s) reloads `latest` and keeps training; without it, the loop exits.

## Architecture

```
┌─────────────────────────────────────┐
│  Platform Layer (src/platform/)     │
│  GameRunner trait + Win32 impl      │
│  Capture (D3D11) + Input (PostMsg)  │
└──────────────┬──────────────────────┘
               │ GameRunner::next_frame() / execute_action()
┌──────────────┴──────────────────────┐
│  Training Layer (platform-agnostic) │
│  FrameStack → ActorCritic → PPO    │
│  GameTracker (per-game rewards)     │
│  Checkpoints + TensorBoard          │
└─────────────────────────────────────┘
```

Two-thread design on Windows (message pump requirement). Training thread uses `GameRunner` trait only — no platform-specific code. Linux port requires implementing `GameRunner` with X11/PipeWire + uinput.

## Reward System (Ninpek)

| Event | Reward | Detection |
|-------|--------|-----------|
| Score increase | +1.0 per 50 points | Score delta from process memory |
| Life gained | +1.0 | Lives delta from process memory |
| Life lost | -1.0 | Lives delta from process memory |
| Stage complete | +10.0 | Stage-clear screen pixel pattern (2-frame stable) |
| Game won (final stage) | +10.0 | Game-complete screen pixel pattern (2-frame stable) |
| Game over | -5.0 | Leaderboard row pattern (2-frame stable) |
| Survival | +0.001/frame | When no other event |

Score (`+0xD0`) and lives (`+0x4C0`) are read from a Game Maker RValue pair on the same object via a shared pointer-chain prefix off `ufo50.exe+0x00742230`. After a reset the tracker waits for `score == 0 && lives == 3` (`observe_idle`) before handing the episode to PPO, so menu animations and the previous run's state never leak into training.

Menu/transition frames (leaderboards, stage-complete, game-complete) are still detected from pixels and excluded from PPO updates via `is_menu`.

## TensorBoard Metrics

| Metric | Description |
|--------|-------------|
| `rollout/ep_rew_mean` | Episode reward |
| `rollout/ep_len_mean` | Episode length (frames) |
| `train/policy_loss` | PPO clipped surrogate loss |
| `train/value_loss` | Value function MSE |
| `train/entropy` | Policy entropy |
| `time/fps` | Training throughput |

## Binaries

| Binary | Description |
|--------|-------------|
| `train_ninpek` | PPO training loop for Ninpek |
| `test_ninpek` | Live reward testing with preview window |
| `test_reset` | Reset-sequence cycle test for tuning input timings |
| `test_model` | Model sanity check (no game needed) |
| `test_train` | PPO smoke test on a synthetic environment |
| `bench_capture` | Capture+downscale FPS benchmark |

## Adding a New Game

1. Create `src/games/yourgame/` with tracker, score, lives, game_over, rewards modules
2. Implement `GameTracker` trait (process_frame, observe_idle, reset_sequence, episode_breakdown, obs/action dims)
3. In `src/games/yourgame/mod.rs`, expose `pub fn definition() -> GameDefinition` with window title, obs dims, action count, tracker factory, and per-game hyperparameters
4. Create `src/bin/train_yourgame.rs` — copy `train_ninpek.rs` and swap `games::ninpek::definition()` for your game's
5. All pixel region coordinates are calibrated per resolution (noted in each game's mod.rs)

No edits to `train::runner`, `platform::host`, or any shared code.

## Project Structure

```
src/
  platform/
    mod.rs              # GameRunner trait, NUM_ACTIONS, ACTION_NAMES, host re-export
    win32/
      mod.rs            # Win32Runner + pub fn host (spawns training on worker)
      capture.rs        # D3D11 GPU capture + downscale + border detection
      input.rs          # 26 discrete actions, vk_noop(ms), reset_game
  games/
    mod.rs              # GameTracker trait, FrameResult, Region struct
    ninpek/
      mod.rs            # pub fn definition() -> GameDefinition
      tracker.rs        # NinpekTracker state machine
      mem.rs            # Process memory reader (score + lives via pointer chain)
      game_over.rs      # Leaderboard, stage/game complete pixel detection
      events.rs         # Event-name string constants shared by tracker + debug
      rewards.rs        # Reward value constants
  train/
    model.rs            # ActorCritic CNN (Nature DQN, parameterized over obs dims)
    ppo.rs              # PPO algorithm + RolloutBuffer + UpdateStats
    preprocess.rs       # FrameStack (BGRA -> grayscale tensor)
    runner.rs           # GameDefinition, Hyperparams, run_training (game-agnostic)
  util/
    cli.rs              # Argument parsing (TrainArgs)
    checkpoint.rs       # Model save/load with JSON metadata, try_load helper
    logger.rs           # TensorBoard logging
  bin/
    train_ninpek.rs     # ~50-line shim: definition() + platform::host + run_training
    test_ninpek.rs      # Reward testing binary with preview window
    test_reset.rs       # Reset-sequence cycle test
    test_model.rs       # Model sanity check
    test_train.rs       # PPO smoke test on synthetic env
    bench_capture.rs    # Capture+downscale FPS benchmark
```
