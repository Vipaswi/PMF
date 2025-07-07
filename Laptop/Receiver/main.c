#include "UDP_receiver.h"
#include "imu_packet.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

volatile motionPacket recentData;

motionPacket* getRecentQuaternionData() {  
    return (motionPacket*)&recentData;  
}

DWORD WINAPI getQuaternionData(LPVOID lpParam)
{
  printf("started ");
  // setup
  startWSA();
  int sock = createSocket();
  int port = 8888;
  if (bind_socket(sock, port) != 0)
  {
    perror("Socket binding failed");
    return -1;
  }

  int tracker = 0;
  motionPacket *buffer = (motionPacket *)malloc(sizeof(motionPacket));

  if (buffer == NULL)
  {
    perror("Buffer allocation failed");
    return -1;
  }

  while (1)
  {
    received_packet(sock, buffer, sizeof(motionPacket));
    memcpy((void*)&recentData, buffer, (size_t)sizeof(motionPacket));
    //printf("Quaternion: w=%.2f x=%.2f y=%.2f z=%.2f\n", buffer->orientData.qw, buffer->orientData.qx, buffer->orientData.qy, buffer->orientData.qz);
  }

  free(buffer);
  closeSocket(sock);
  closeWSA();

  return 0;
}

int startQuaternionThread() {
    if (CreateThread(NULL, 0, getQuaternionData, NULL, 0, NULL) == NULL) {
        perror("Thread creation failed");
    };
    
    return 0;
}