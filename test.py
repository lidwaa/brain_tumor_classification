import tensorflow as tf

# Vérifiez si un GPU est disponible
if tf.config.list_physical_devices('GPU'):
    print("GPU est disponible.")
else:
    print("GPU n'est pas disponible.")
