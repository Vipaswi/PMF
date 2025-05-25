// imu_packet.h
typedef struct {
    float ax, ay, az; //angular accel
    float gx, gy, gz; //angular vel.
    float mx, my, mz; //mag. field (not relevant?)
} IMUPacket; //a singular IMU packet defined by vectors

/**
 * @brief This quaternion is sent over to the source/laptop through 
 * UDP, and converted into Euler angles to be rendered later.
 */
typedef struct {
    float qw, qx, qy, qz;
} Quaternion; 
