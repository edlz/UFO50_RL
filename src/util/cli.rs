pub struct TrainArgs {
    pub namespace: String,
    pub max_episodes: Option<u32>,
    pub max_frames: Option<u64>,
    pub max_minutes: Option<u64>,
    pub auto_resume: bool,
    pub debug: bool,
}

pub fn parse_train_args(default_namespace: &str) -> TrainArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut namespace = default_namespace.to_string();
    let mut max_episodes = None;
    let mut max_frames = None;
    let mut max_minutes = None;
    let mut auto_resume = false;
    let mut debug = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--namespace" | "-n" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    namespace = val.clone();
                }
            }
            "--episodes" | "-e" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    max_episodes = val.parse().ok();
                }
            }
            "--frames" | "-f" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    max_frames = val.parse().ok();
                }
            }
            "--minutes" | "-m" => {
                i += 1;
                if let Some(val) = args.get(i) {
                    max_minutes = val.parse().ok();
                }
            }
            "--auto-resume" | "-r" => auto_resume = true,
            "--debug" | "-d" => debug = true,
            _ => {}
        }
        i += 1;
    }
    TrainArgs {
        namespace,
        max_episodes,
        max_frames,
        max_minutes,
        auto_resume,
        debug,
    }
}
