

int main() {
    int sock = create_socket();
    if (sock < 0) return 1;

    struct IMUPacket packet = ...; // fill this in from your IMU

    while (1) {
        send_packet(sock, "192.168.1.50", 8888, &packet, sizeof(packet));
        Sleep(10); // milliseconds between packets
    }

    closesocket(sock);
    WSACleanup();
    return 0;
}
