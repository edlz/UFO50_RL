# ufo50ppo

PPO reinforcement learning agent that learns to play [UFO 50](https://50games.fun/) games by capturing the screen and sending keyboard inputs. Windows-only (Linux-portable architecture).

Currently trains on **Ninpek** (game 3) with pixel-based reward detection for score, lives, stage completion, and game over.

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

# Train with limits
cargo run --release --bin train_ninpek -- -e 100 -f 500000

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
| `--debug` | `-d` | Save frames to `debug_frames/ep_NNNN/`, print per-episode reward breakdown |

## Training Output

```
checkpoints/ninpek/{namespace}/
  latest.safetensors          # most recent model (+ .json metadata)
  best.safetensors            # highest episode reward
  update_000042.safetensors   # versioned checkpoints (every 50k frames)

runs/ninpek/{namespace}/
  20260405_143022/            # tensorboard logs (timestamped per run)
```

Resuming training automatically loads from `latest.safetensors` and continues from the saved episode/frame/update count.

Episode timeout (60s) rolls back to last versioned checkpoint and exits.

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
| Score increase | +1.0 | B/W pixel flips in score region (quantized, 2-frame stable) |
| Life gained | +1.0 | Blue pixel slot counting, boundary-only check, 2-frame stable |
| Life lost | -1.0 | Same as above |
| Stage complete | +10.0 | Blue+orange icons + center white text + black screen |
| Game over | -5.0 | Leaderboard row pattern detection |
| Survival | +0.001/frame | Always, when no other event |

Menu/transition frames (black screens, game selection, leaderboards) are automatically detected and skipped.

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
| `bench_capture` | Capture performance benchmark |
| `test_model` | Model sanity check (no game needed) |

## Adding a New Game

1. Create `src/games/yourgame/` with tracker, score, lives, game_over, rewards modules
2. Implement `GameTracker` trait (process_frame, is_menu_screen, extra_reset_keys, config)
3. Create `src/bin/train_yourgame.rs` — swap `NinpekTracker::new()` for your tracker
4. All pixel region coordinates are calibrated per resolution (noted in each game's mod.rs)

## Project Structure

```
src/
  platform/
    mod.rs              # GameRunner trait, NUM_ACTIONS, ACTION_NAMES
    win32/
      mod.rs            # Win32Runner (channels + message pump)
      capture.rs        # D3D11 GPU capture + downscale + border detection
      input.rs          # 26 discrete actions via PostMessage
  games/
    mod.rs              # GameTracker trait, FrameResult, Region struct
    ninpek/
      tracker.rs        # NinpekTracker state machine (implements GameTracker)
      score.rs          # Score OCR (quantized B/W classification)
      lives.rs          # Slot-based life counting
      game_over.rs      # Leaderboard, stage/game complete, menu detection
      rewards.rs        # Reward value constants
  train/
    model.rs            # ActorCritic CNN (Nature DQN architecture)
    ppo.rs              # PPO algorithm + RolloutBuffer + UpdateStats
    preprocess.rs       # FrameStack (BGRA -> grayscale tensor)
  util/
    cli.rs              # Argument parsing (TrainArgs)
    checkpoint.rs       # Model save/load with JSON metadata
    logger.rs           # TensorBoard logging
  bin/
    train_ninpek.rs     # Training binary
    test_ninpek.rs      # Reward testing binary
    bench_capture.rs    # Capture benchmark
    test_model.rs       # Model sanity check
```
