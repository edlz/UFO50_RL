# ufo50ppo

PPO reinforcement learning agent that learns to play [UFO 50](https://50games.fun/) games by capturing the screen and sending keyboard inputs. Windows-only.

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

# Debug mode (saves every frame as BMP)
cargo run --release --bin train_ninpek -- -d
```

## Training Output

```
checkpoints/ninpek/{namespace}/
  latest.safetensors    # most recent model
  latest.json           # metadata (episode, frames, updates, best reward)
  best.safetensors      # highest episode reward
  best.json
  update_000042.safetensors  # versioned checkpoints (every 50k frames)
  update_000042.json

runs/ninpek/{namespace}/
  20260405_143022/       # tensorboard logs (timestamped per run)
```

View training progress:
```bash
tensorboard --logdir runs/ninpek
```

## Binaries

| Binary | Description |
|--------|-------------|
| `train_ninpek` | PPO training loop for Ninpek |
| `test_ninpek` | Live reward testing with preview window |
| `bench_capture` | Capture performance benchmark |
| `test_model` | Model sanity check (no game needed) |

## CLI Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--namespace` | `-n` | Training namespace (default: "default") |
| `--episodes` | `-e` | Max episodes before stopping |
| `--frames` | `-f` | Max frames before stopping |
| `--debug` | `-d` | Save every frame to `debug_frames/ep_NNNN/` |

## Architecture

Two-thread design:

- **Capture thread** (main): Win32 message pump, GPU screen capture via Windows.Graphics.Capture, D3D11 downscale to 128x128, auto border crop
- **Training thread**: PPO training loop with NinpekTracker reward system

```
Game Window -> GPU Capture -> 128x128 BGRA -> [channel] -> FrameStack -> ActorCritic -> action -> [channel] -> PostMessage -> Game
```

## Reward System (Ninpek)

| Event | Reward | Detection |
|-------|--------|-----------|
| Score increase | +1.0 | B/W pixel flips in score region (quantized, 2-frame stable) |
| Life gained | +1.0 | Blue pixel slot counting, boundary-only check, 2-frame stable |
| Life lost | -1.0 | Same as above |
| Stage complete | +10.0 | Blue+orange icons + center white text + black screen |
| Game over | -5.0 | Leaderboard row pattern detection |
| Survival | +0.01/frame | Always, when no other event |

Episode timeout (60s) rolls back to last versioned checkpoint and exits.

## Project Structure

```
src/
  game/
    capture.rs      # D3D11 GPU capture + downscale + border detection
    input.rs        # 26 discrete actions via PostMessage
    games/
      ninpek/
        tracker.rs  # NinpekTracker state machine
        score.rs    # Score OCR (quantized B/W classification)
        lives.rs    # Slot-based life counting
        game_over.rs # Leaderboard + stage complete detection
        rewards.rs  # Reward value constants
  train/
    model.rs        # ActorCritic CNN (Nature DQN architecture)
    ppo.rs          # PPO algorithm + RolloutBuffer
    preprocess.rs   # FrameStack (BGRA -> grayscale tensor)
  util/
    cli.rs          # Argument parsing
    checkpoint.rs   # Model save/load with metadata
    logger.rs       # TensorBoard logging
  bin/
    train_ninpek.rs # Training binary
    test_ninpek.rs  # Reward testing binary
    bench_capture.rs
    test_model.rs
```
