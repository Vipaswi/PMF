/**
 * @file UDP_receiver.h
 * @author Vipaswi Thapa (vt7637@g.rit.edu)
 * @brief 
 * @version 0.1
 * @date 2025-05-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <stdio.h> //for size_t
#include "UDP_export.h"
#include "imu_packet.h"

DLL_EXPORT const motionPacket* getLatestPacket();

DLL_EXPORT int startWSA();
DLL_EXPORT int closeWSA();
DLL_EXPORT int createSocket();
DLL_EXPORT int bind_socket(int sock, int port);
DLL_EXPORT int received_packet(int sock, void* buffer, size_t size);
DLL_EXPORT int closeSocket(int sock);