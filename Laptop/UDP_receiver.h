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

int startWSA();
int closeWSA();
int createSocket();
int bind_socket(int sock, int port);
int received_packet(int sock, void* buffer, size_t size);
int closeSocket(int sock);