// Only marker emission is feature-gated; native marker calls are no-ops.
macro_rules! profile_start {
    ($label:literal) => {
        #[cfg(all(feature = "markers", target_os = "zkvm"))]
        println!(concat!("cycle-tracker-report-start: ", $label));
    };
}
macro_rules! profile_end {
    ($label:literal) => {
        #[cfg(all(feature = "markers", target_os = "zkvm"))]
        println!(concat!("cycle-tracker-report-end: ", $label));
    };
}
macro_rules! profile_calibrate {
    () => {
        #[cfg(all(feature = "markers", target_os = "zkvm"))]
        for _ in 0..32 {
            profile_start!("empty_marker");
            profile_end!("empty_marker");
        }
    };
}
