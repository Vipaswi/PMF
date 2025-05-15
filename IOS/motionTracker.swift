import CoreMotion

let motionManager = CMMotionManager()

func startMotionUpdates() {
    if motionManager.isDeviceMotionAvailable {
        motionManager.deviceMotionUpdateInterval = 1.0 / 100.0 // 100 Hz
        motionManager.startDeviceMotionUpdates(to: .main) { (motion, error) in
            guard let motion = motion else { return }
            let attitude = motion.attitude
            let quat = attitude.quaternion
            
            // Now send this data over UDP socket (you'll bridge to your C code here)
        }
    }
}