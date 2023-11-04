def delta_compress(data):
    compressed_data = bytearray()
    previous_byte = 0

    for byte in data:
        diff = byte - previous_byte
        if diff < 0:
            diff += 256
        compressed_data.append(diff)
        previous_byte = byte

    return compressed_data


def delta_decompress(compressed_data):
    data = bytearray()
    previous_byte = 0

    for diff in compressed_data:
        byte = previous_byte + diff
        if byte > 255:
            byte -= 256
        data.append(byte)
        previous_byte = byte

    return data