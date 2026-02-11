import mmap
import struct
import numpy as np
import json
import ctypes
from ctypes import wintypes

data = np.zeros(19, dtype=np.float32)

with open('temp_data/address.json', 'r') as file:
    address_hex = json.load(file)
address_int = int(address_hex, 16);
print("data to be fetched from " + address_hex + " / " + str(address_int))

def fetch():
    # On Windows, mmap(-1, ...) is used for anonymous memory mapping,
    # but when a name is provided it uses the shared memory mapping.
    # We use the tag= parameter to specify the name on Windows.
    # On Unix, the name would be the file path to the mmap file descriptor.
    try:
        # Open the existing memory-mapped file
        # 'tag' is used for the name of the shared memory region on Windows
        mm = mmap.mmap(-1, 1024, tagname="DataMap", access=mmap.ACCESS_READ)

        # Read the message length (4 bytes at offset 0)
        # 'i' means integer (4 bytes)
        length = struct.unpack('i', mm[:4])[0]

        # Read the actual message bytes from offset 4 to 4 + length
        message_bytes = mm[4 : 4 + length]
        message = message_bytes.decode('utf-8')

        print(f"Python read message: '{message}'")

        # Close the memory map
        mm.close()

    except Exception as e:
        print(f"Error accessing shared memory: {e}")
        print("Ensure the C# program is running first.")
    
fetch()