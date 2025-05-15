/**
 * Initialize Winsock.
Create a socket.
Bind the socket.
Listen on the socket for a client.
Accept a connection from a client.
Receive and send data.
Disconnect.
 */


 
int bind_socket(int port); //
int received_packet(int sock, void* buffer, size_t size);