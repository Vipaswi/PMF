/**
 * @brief The UDP transmitter, repsonsible for sending the received IMU packet data.
 * 
 */
#include "imu_packet.h"

#include <unistd.h>         // close()
#include <sys/types.h>      // type size_t
#include <sys/socket.h>     // socket(), sendto(), recvfrom()
#include <netinet/in.h>     // for sockaddr_in
#include <arpa/inet.h>      // for inet_pton(), htons(), etc.
#include <string.h> //memset
#include <stdio.h> //perror

#define dest_ip "192.168.1.129"
#define dest_port 8888

/**
 * @brief Create a Socket object
 * 
 * @return the socket; negative if failure.
 */
int createSocket(void){
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    return sock;
}

/**
 * @brief Transmit a single IMUPacket over the socket
 *
 * @param sock the socket id
 * @param packet the packet address
 *
 */
long transmitPacket(int sock, motionPacket* packet) {
    //Define destination address:
    struct sockaddr_in destAddr;
    memset(&destAddr, 0, sizeof(destAddr)); //populate memory with 0s

    destAddr.sin_family = AF_INET; //IPV4 address family
    destAddr.sin_port = htons(dest_port); //network byte order conversion from port

    //Internet presentation to network; IPV4 string -> string ip address into destAddr.sin_addr format (binary).
    if (inet_pton(AF_INET, dest_ip, &destAddr.sin_addr) <= 0) { //IP-string to binary format
        perror("Invalid IP address");
        return -1;
    }

    ssize_t sent = sendto(sock, packet, sizeof(*packet), 0,
                          (struct sockaddr*)&destAddr, sizeof(destAddr)); //signed sizes that may exceed int limits
    
    return sent;
}

/**
 * @brief Closes the socket and cleans it up
 * 
 * @param sock the socket to be closed
 */
void closeSocket(int sock){
    close(sock);
}
