import struct

def rle_encode(data):
    encoded_data = bytearray()
    i = 0
    while i < len(data):
        count = 1
        while i + count < len(data) and data[i+count] == data[i]:
            count += 1
        if count == 1:
            encoded_data += struct.pack('>B', data[i])
        else:
            encoded_data += struct.pack('>BI', 0, count) + struct.pack('>B', data[i])
        i += count
    return encoded_data

def rle_decode(encoded_data):
    decoded_data = bytearray()
    i = 0
    while i < len(encoded_data):
        flag = encoded_data[i]
        if flag == 0:
            count = struct.unpack('>I', encoded_data[i+1:i+5])[0]
            value = encoded_data[i+5]
            decoded_data.extend(bytes([value] * count))
            i += 6
        else:
            decoded_data.append(flag)
            i += 1
        print(i)
    print('Decoding!!!!!!')
    return decoded_data

def rle_compress(data):
    count = 1
    value = data[0]
    output = []
    for byte in data[1:]:
        if byte == value and count < 65535:
            count += 1
        else:
            while count > 256:
                output.append(255)
                output.append(value)
                count -= 256
            output.append(count-1)
            output.append(value)
            count = 1
            value = byte
    while count > 256:
        output.append(255)
        output.append(value)
        count -= 256
    output.append(count-1)
    output.append(value)
    return bytes(output)

def rle_decompress(compressed_data):
    output = []
    count = 0
    for i, byte in enumerate(compressed_data):
        if i % 2 == 0:
            count = byte
            if count == 255:
                count = 256 + compressed_data[i+1]
        else:
            output.extend([byte] * (count+1))
    return bytes(output)

