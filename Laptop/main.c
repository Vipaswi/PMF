#pragma once
#include "UDP_receiver.h"
#include "imu_packet.h"

void main(){
  //setup
  startWSA();
  int sock = createSocket();
  int port = 8888;
  bind_socket(sock, port);
  

  int tracker = 0;

  while(){

  }

  //disconnect
}


