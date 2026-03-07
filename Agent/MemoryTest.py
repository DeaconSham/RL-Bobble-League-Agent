import SharedMemoryLink
import numpy as np

mem_link = SharedMemoryLink.SharedMemoryLink()

print(mem_link.read_observation())