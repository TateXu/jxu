import socket
import sys
import numpy as np

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the port
server_address = ('localhost', 10000)
print >>sys.stderr, 'starting up on %s port %s' % server_address
sock.bind(server_address)
fs = 512
epochsample = 32
nchn = 69
tlim = np.int(1.0 / epochsample * fs)

data_buffer = np.empty((nchn, 0))
# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print >>sys.stderr, 'waiting for a connection'
    connection, client_address = sock.accept()
    try:
        print >>sys.stderr, 'connection from', client_address

        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(10000)
            if data:
                true_data = np.fromstring(data[1:], sep=' ')

                if data_buffer.shape[1] < tlim:
                    data_buffer = np.append(data_buffer, np.asarray([true_data]).T, axis=1)
                else:
                    print data_buffer.shape
                    data_buffer = np.empty((nchn, 0))
                # print >>sys.stderr, 'received "%s"' % data
                # print >>sys.stderr, 'sending data back to the client'
                connection.sendall(data)
            else:
                print >>sys.stderr, 'no more data from', client_address
                break


    finally:
        # Clean up the connection
        connection.close()
        
