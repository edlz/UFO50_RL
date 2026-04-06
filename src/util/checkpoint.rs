use std::io::{Read, Write};

pub struct CheckpointMeta {
    pub game: &'static str,
    pub resolution: (u32, u32),
    pub episode: u32,
    pub total_frames: u64,
    pub ppo_updates: u64,
    pub best_reward: f64,
    pub rollout_len: usize,
    pub learning_rate: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
}

pub fn save_metadata(dir: &str, name: &str, meta: &CheckpointMeta) {
    let path = format!("{}/{}.json", dir, name);
    if let Ok(mut f) = std::fs::File::create(&path) {
        let _ = write!(
            f,
            concat!(
                "{{\n",
                "  \"game\": \"{}\",\n",
                "  \"resolution\": [{}, {}],\n",
                "  \"episode\": {},\n",
                "  \"total_frames\": {},\n",
                "  \"ppo_updates\": {},\n",
                "  \"best_reward\": {:.1},\n",
                "  \"rollout_len\": {},\n",
                "  \"learning_rate\": {},\n",
                "  \"gamma\": {},\n",
                "  \"gae_lambda\": {}\n",
                "}}"
            ),
            meta.game,
            meta.resolution.0,
            meta.resolution.1,
            meta.episode,
            meta.total_frames,
            meta.ppo_updates,
            meta.best_reward,
            meta.rollout_len,
            meta.learning_rate,
            meta.gamma,
            meta.gae_lambda
        );
    }
}

/// State loaded from a previous checkpoint to resume training.
pub struct ResumedState {
    pub episode: u32,
    pub total_frames: u64,
    pub ppo_updates: u64,
    pub best_reward: f64,
}

/// Try to load metadata from a checkpoint JSON file.
/// Returns None if file doesn't exist or can't be parsed.
pub fn load_metadata(dir: &str, name: &str) -> Option<ResumedState> {
    let path = format!("{}/{}.json", dir, name);
    let mut contents = String::new();
    std::fs::File::open(&path)
        .ok()?
        .read_to_string(&mut contents)
        .ok()?;

    // Simple JSON parsing without serde — extract numeric fields
    fn extract_u32(s: &str, key: &str) -> Option<u32> {
        let pat = format!("\"{}\":", key);
        let idx = s.find(&pat)? + pat.len();
        s[idx..]
            .trim()
            .split(|c: char| !c.is_ascii_digit())
            .next()?
            .parse()
            .ok()
    }
    fn extract_u64(s: &str, key: &str) -> Option<u64> {
        let pat = format!("\"{}\":", key);
        let idx = s.find(&pat)? + pat.len();
        s[idx..]
            .trim()
            .split(|c: char| !c.is_ascii_digit())
            .next()?
            .parse()
            .ok()
    }
    fn extract_f64(s: &str, key: &str) -> Option<f64> {
        let pat = format!("\"{}\":", key);
        let idx = s.find(&pat)? + pat.len();
        s[idx..]
            .trim()
            .split(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
            .next()?
            .parse()
            .ok()
    }

    Some(ResumedState {
        episode: extract_u32(&contents, "episode")?,
        total_frames: extract_u64(&contents, "total_frames")?,
        ppo_updates: extract_u64(&contents, "ppo_updates")?,
        best_reward: extract_f64(&contents, "best_reward").unwrap_or(f64::NEG_INFINITY),
    })
}
