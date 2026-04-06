use ufo50ppo::train;

fn main() {
    // Pre-load torch_cuda.dll
    unsafe {
        windows::Win32::System::LibraryLoader::LoadLibraryA(windows::core::s!("torch_cuda.dll"))
            .ok();
    }

    let device = tch::Device::cuda_if_available();
    println!("Device: {:?}", device);

    let model = train::model::ActorCritic::new(device);
    println!("Model created");

    // Dummy observation: [1, 4, 128, 128]
    let obs = tch::Tensor::randn([1, 4, 128, 128], (tch::Kind::Float, device));
    println!("Obs shape: {:?}", obs.size());

    // Forward pass
    let (log_probs, values) = model.forward(&obs);
    println!("Log probs shape: {:?}", log_probs.size());
    println!("Values shape: {:?}", values.size());
    println!("Log probs: {:?}", log_probs);

    // Sample action
    let (action, log_prob, value) = model.act(&obs);
    println!(
        "Action: {}, log_prob: {:.4}, value: {:.4}",
        action, log_prob, value
    );

    // Test frame stack
    let mut fs = train::preprocess::FrameStack::new(device);
    let fake_bgra = vec![128u8; 128 * 128 * 4];
    let stacked = fs.push(&fake_bgra, 128, 128);
    println!("Frame stack output: {:?}", stacked.size());

    // Test act on stacked output
    let (action2, _, _) = model.act(&stacked);
    println!("Action from frame stack: {}", action2);

    println!("\nAll checks passed!");
}
