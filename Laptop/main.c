#include "UDP_receiver.h"
#include "imu_packet.h"
#include "stdio.h"

int main(){
  //setup
  startWSA();
  int sock = createSocket();
  int port = 8888;
  if(bind_socket(sock, port) != 0){
    perror("Socket binding failed");
    return -1;
  }
  
  int tracker = 0;
  Quaternion* buffer = (Quaternion*) malloc(sizeof(Quaternion));

  if(buffer == NULL){
    perror("Buffer allocation failed");
    return -1;
  }

  while(1){
    received_packet(sock, buffer, sizeof(Quaternion));
    printf("Quaternion: w=%.2f x=%.2f y=%.2f z=%.2f\n", buffer->qw, buffer->qx, buffer->qy, buffer->qz);
  }

  free(buffer);
  closeSocket(sock);
  closeWSA();

  return 0;
}


