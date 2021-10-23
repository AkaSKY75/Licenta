import socket, ssl

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5)

f = open("C:\\Users\\user\\Desktop\\result.txt", "w")

ss = ssl.wrap_socket(s, ssl_version=ssl.PROTOCOL_TLSv1)

ADDR = "www.youtube.com"
request="GET / HTTP/1.1\r\nHost: "+ADDR+"\r\n\r\n"

ss.connect((ADDR, 443))
ss.sendall(request.encode())

bytes = b''

while True:
    try:
        data = ss.recv(1)
        bytes += data
        if data == b'\n':
            f.write("\n")
        if int.from_bytes(data, "big") > 32 and int.from_bytes(data, "big") < 127:
            f.write(data.decode("utf-8"))
    except Exception as e:
        print(e)
        break
print(bytes)
f.close()
