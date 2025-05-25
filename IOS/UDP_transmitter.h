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
 int transmitPacket(int sock, struct IMUPacket* packet);
 void closeSocket();

 #ifdef __cplusplus
}
#endif