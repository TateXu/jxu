import socket
import sys
import time
import numpy as np
from loaddata import loadeeg
# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('localhost', 10000)
print >>sys.stderr, 'connecting to %s port %s' % server_address

fs = 512
datamat = loadeeg()
if datamat.ndim != 2:
    raise ValueError("INPUT must be 2-dim!")
len_off_data = datamat.shape[1]
sock.connect(server_address)
try:
    # Send data
    # amount_expected = len(message)
    counter = 0
    while True: ## amount_received < amount_expected:
        mysample = np.array_str(datamat[:, counter].ravel())
        # mysample = datamat[:, counter].ravel().tolist()
        # str_sp = ' '.join(map(str, mysample))
        # now send it and wait

        #  message = 'This is the message.  It will be repeated.'
        #  print >>sys.stderr, 'sending "%s"' % message
        # print >>sys.stderr, 'sending "%s"' %mysample

        sock.sendall(mysample)
        data = sock.recv(10000)
        if mysample != data:
            # raise ValueError("Error!!!") 
            pass
        else:
            print "True"
        # print >>sys.stderr, 'received "%s"' % data
        counter = 0 if counter == len_off_data - 1 else counter + 1
        time.sleep(1.0 / fs)
finally:
    print >>sys.stderr, 'closing socket'
    sock.close()
