/**
 * @file UDP_transmitter.h
 * @author Vipaswi Thapa (vt7637@g.rit.edu)
 * @brief 
 * @version 0.1
 * @date 2025-05-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once
#include "imu_packet.h"

#ifdef __cplusplus
extern "C" {
#endif


 int createSocket();
 long transmitPacket(int sock, motionPacket* packet);
 void closeSocket(int sock);

 #ifdef __cplusplus
}
#endif
