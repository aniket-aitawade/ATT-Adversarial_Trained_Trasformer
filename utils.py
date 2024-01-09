import tensorflow as tf

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def LR_scheduler_callback():
   return tf.keras.callbacks.LearningRateScheduler( scheduler, verbose=1)

def early_stop_callback():
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_total g loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
    )

def model_checkpoints_callback():
   return tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/',
    verbose=1,
    save_weights_only=True,
    monitor='val_total g loss',
    mode='min',
    save_best_only=True)

def tensorboard_callback():
   return tf.keras.callbacks.TensorBoard(log_dir="./logs")

if __name__=="__main__":
  pass