from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from tensorflow.python.client import device_lib

# List local devices
local_devices = device_lib.list_local_devices()

# Print local devices
for device in local_devices:
    print(device.name, device.device_type, device.memory_limit)