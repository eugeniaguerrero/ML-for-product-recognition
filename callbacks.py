import keras
from keras.callbacks import LambdaCallback
import json

class logs(object):
    def __init__(self):
        # Print the batch number at the beginning of every batch.
        self.batch_print_callback = LambdaCallback(
            on_batch_begin=lambda batch, logs: print(batch))

        # Stream the epoch loss to a file in JSON format. The file content
        # is not well-formed JSON but rather has a JSON object per line.

        self.json_log = open('loss_log.json', mode='wt', buffering=1)
        self.json_logging_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: self.json_log.write(
                json.dumps({'epoch': epoch, 'loss': logs['loss'], 'accuracy' : logs['acc']}) + '\n'),
            on_train_end=lambda logs: self.json_log.close()
        )

        # Terminate some processes after having finished model training.
        processes = ...
        cleanup_callback = LambdaCallback(
            on_train_end=lambda logs: [
                p.terminate() for p in processes if p.is_alive()])
