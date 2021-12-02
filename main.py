import tensorflow as tf

from model.unet import UNet


if __name__ == '__main__':

    """ TF / GPU config """
    tf.random.set_seed(1234)
    tf.keras.backend.clear_session()
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = InteractiveSession(config=config)

    model = UNet()

    model.load_data()

    model.build()
    model.callbacks()
    model.train()

    model.get_best_model()
    model.evaluate()
    model.see_random_results()
    model.get_metrics()
    model.log_extreme_outputs()
    model.log()

    model.model_compare()
    model.predict()

    # TODO Three workflows:
    #  1. Training and logging to tensorflow
    #  2. Model loading and evaluation
    #  3. Test on simulation and experiment data
