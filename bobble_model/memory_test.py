import shared_memory_link
import numpy as np

mem_link = shared_memory_link.Shared_memory_link()

print(mem_link.read_observation())