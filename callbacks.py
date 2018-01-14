import keras
from keras.callbacks import LambdaCallback
import json
import requests


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
                json.dumps({'epoch': epoch, 'loss': logs['loss'], 'accuracy': logs['acc']}) + '\n'),
            on_train_end=lambda logs: self.json_log.close()
        )

        self.slack_callback = LambdaCallback(
            on_train_begin=lambda logs: self.message(logs)
        )


    def message(self,logs):
        slack_data = {'text': "The wolf has started to prowl"}
        webhook_url = 'https://hooks.slack.com/services/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'

        response = requests.post(
            webhook_url, data=json.dumps(slack_data),
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code != 200:
            raise ValueError(
                'Request to slack returned an error %s, the response is:\n%s'
                % (response.status_code, response.text)
            )

"""
    def start_training_notification(self):
        self.batch_print_callback = LambdaCallback(
            on_train_begin=lambda batch, logs: print(slack_data))



    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if not np.isnan(loss) or not np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True

        # Terminate some processes after having finished model training.
        processes = ...
        cleanup_callback = LambdaCallback(
            on_train_end=lambda logs: [
                p.terminate() for p in processes if p.is_alive()])"""

