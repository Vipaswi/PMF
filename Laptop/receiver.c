#include <imu_packet.h>
#include <stdio.h> //printing
#include <stdlib.h>             // processing/controlling memory and memory
#include <string.h>             // memory deets: memset, memcpy
#include <unistd.h>             // For close (Unix-like systems)
#include <winsock2.h>
#include <ws2tcpip.h>

int sock = bind_socket(8888);

IMUPacket packet;
while (1) {
    receive_packet(sock, &packet, sizeof(packet));
    printf("Accel: %.2f %.2f %.2f\n", packet.ax, packet.ay, packet.az);
    // Optional: run Madgwick + update OpenGL
}
