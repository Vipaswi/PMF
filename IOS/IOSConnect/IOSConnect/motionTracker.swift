import CoreMotion

let motionManager = CMMotionManager()

func startMotionUpdates(_ socket: Int32) {
    if motionManager.isDeviceMotionAvailable {
        motionManager.deviceMotionUpdateInterval = 1.0 / 100.0 // 100 Hz
        motionManager.startDeviceMotionUpdates(to: .main) { (motion, error) in
            guard let motion = motion else { return }
            let attitude = motion.attitude
            let quat = attitude.quaternion
            
            // Now send this data over UDP socket (you'll bridge to your C code here)
            // Initialize a C Quaternion struct
            let orient = Quaternion(qw: Float(quat.w),
                                  qx: Float(quat.x),
                                  qy: Float(quat.y),
                                  qz: Float(quat.z))
            
            let accel = motion.userAcceleration;
            let grav = motion.gravity;
            let rotRate = motion.rotationRate;
            
            let imu = IMUPacket(ax: Float(accel.x), ay: Float(accel.y), az: Float(accel.z),
                                gx: Float(grav.x), gy: Float(grav.y), gz: Float(grav.z),
                                mx: Float(rotRate.x), my: Float(rotRate.y), mz: Float(rotRate.z));
            
            var motionPack = motionPacket(rawData: imu, orientData: orient);

            // Transmit it over UDP using C function
            let sent = transmitPacket(socket, &motionPack);
            if sent < 0 {
                print("Failed to send motion data")
            } else {
                print("Motion data sent: Accelx: " + String(accel.x));
            }
            
        }
    } else {
        print("Motion data not available for this device!")
    }
}
