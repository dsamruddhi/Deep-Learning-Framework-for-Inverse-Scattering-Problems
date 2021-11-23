from model.unet import UNet


if __name__ == '__main__':

    model = UNet()
    model.load_data()
    model.build()
    model.checkpoint()
    model.tensorboard()
    model.train()
    model.evaluate()
