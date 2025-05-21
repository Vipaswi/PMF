/**
 * Initialize Winsock.
Create a socket.
Bind the socket.
Listen on the socket for a client.
Accept a connection from a client.
Receive and send data.
Disconnect.
 */

#include <imu_packet.h>
#include <stdio.h> //printing
#include <stdlib.h>             // processing/controlling memory and memory
#include <string.h>             // memory deets: memset, memcpy
#include <unistd.h>             // For close (Unix-like systems)
#include <winsock2.h>
#include <ws2tcpip.h>

/**
 * @brief Starts a winsock instance
 * 
 * @return int : the WSA
 */
int startWSA(){
  WSADATA wsadata; //declare WSAData structure
  return WSAStartup(MAKEWORD(2,2), &wsadata); //start up the socket
}

int closeWSA(){
  return WSACleanup();
}

/**
 * @brief Create a Socket object. 
 * 
 * @return -1 for failure, else the socket
 */
int createSocket(){
  int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  return sock == INVALID_SOCKET ? -1 : sock; //No error handling as of yet
}

/**
 * @brief close socket
 * 
 * @return -1 for failure, else 0
 */
int closeSocket(int sock){
  return closesocket(sock);
}

/**
 * @brief binds a socket so it listens to a port's UDP datagrams
 * 
 * @param sock : the socket id
 * @param port : the port id
 * @return int : the bind success (0 on success, -1 on err)
 */
int bind_socket(int sock, int port){
  //initialize
  struct sockaddr_in serverAddr; //Creates IPv4 addr. struct
  
  //define
  memset(&serverAddr, 0, sizeof(serverAddr));       //clear memory (JIC)
  serverAddr.sin_family = AF_INET;                  //define addr. family
  serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);   //IP addr. : INADDR_ANY binds to all local interfaces
  serverAddr.sin_port = htons(port);                //The # of the port, where htons puts it in network byte order
  
  //bind:
  return bind(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)); //bind the socket to the sockAddress pointer with its given. size (based on sockAddr_in struct)
}
 
/**
 * @brief Receive a UDP packet
 * 
 * @param sock Socket descriptor
 * @param buffer Pointer to where data will be stored
 * @param size Size of expected data
 * @return int Number of bytes received or -1 on failure
 */
int received_packet(int sock, void* buffer, size_t size){
    struct sockaddr_in clientAddr;
    int addrLen = sizeof(clientAddr);

    int bytesReceived = recvfrom(sock, (char*)buffer, size, 0,
                                 (struct sockaddr*)&clientAddr, &addrLen);

    return (bytesReceived == SOCKET_ERROR) ? -1 : bytesReceived;
}