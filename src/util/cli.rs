pub struct TrainArgs {
    pub namespace: String,
    pub max_episodes: Option<u32>,
    pub max_frames: Option<u64>,
    pub debug: bool,
}

pub fn parse_train_args(default_namespace: &str) -> TrainArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut namespace = default_namespace.to_string();
    let mut max_episodes = None;
    let mut max_frames = None;
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
            "--debug" | "-d" => debug = true,
            _ => {}
        }
        i += 1;
    }
    TrainArgs {
        namespace,
        max_episodes,
        max_frames,
        debug,
    }
}
