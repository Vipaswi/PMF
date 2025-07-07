#ifndef IMU_PACKET_H
#define IMU_PACKET_H

#pragma pack(push, 1)

// imu_packet.h
typedef struct {
    float ax, ay, az; //angular accel
    float gx, gy, gz; //angular vel.
    float mx, my, mz; //mag. field (not relevant?)
} IMUPacket; //a singular IMU packet defined by vectors

/**
 * @brief This quaternion is sent over to the source/laptop through 
 * UDP, and converted into Euler angles to be rendered in OPENGL.
 */
typedef struct {
    float qw, qx, qy, qz;
} Quaternion; 

/**
 * @brief Packet of data that contains both raw data (accel, velocity, and magn. field data),
 *        and orientation data.
 * 
 */
typedef struct{
    IMUPacket rawData; //raw acceleration and velocity data
    Quaternion orientData; //the orientation data
} motionPacket;


#pragma pack(pop)
#endif 