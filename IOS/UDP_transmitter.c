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

/**
 * @brief Create a Socket object
 * 
 * @return the socket
 */
int createSocket(){
    int sock = create_socket();
    return sock;
}

/**
 * @brief Transmit a single IMUPacket over the socket
 * 
 * @param packet 
 */
int transmitPacket(int sock, struct IMUPacket* packet) {
    
    //return 1 if failure
    if (sock < 0) {
        return 1;
    }

    
    send_packet(sock, "192.168.1.50", 8888, &packet, sizeof(packet));
    Sleep(10); // milliseconds between packets

    
    return 0;
}

/**
 * @brief Closes the socket and cleans it up
 * 
 * @param sock: the socket to be closed
 */
void closeSocket(int sock){
    closesocket(sock);
    WSACleanup();
}
