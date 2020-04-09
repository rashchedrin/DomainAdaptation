import numpy as np
from loss import loss_DANN
from models import DANNModel
import configs.dann_config as dann_config


def random_batch(
        image_shape=(70, 81),
        n_images_src=12,
        n_images_trg=5,
        n_classes=10,
        n_channels=3,
        unknown_proportion=0.95):
    batch = dict()
    batch['src_images'] = np.random.randn(n_images_src, n_channels, *image_shape)
    batch['trg_images'] = np.random.randn(n_images_trg, n_channels, *image_shape)
    batch['src_classes'] = np.random.randint(0, n_classes, n_images_src)
    batch['trg_classes'] = np.random.randint(0, n_classes, n_images_trg)
    unk_mask = np.random.random(n_images_trg) < unknown_proportion
    batch['trg_classes'][unk_mask] = dann_config.UNK_VALUE
    return batch


def test_loss_DANN():
    """ Loss and model should work together """

    model = DANNModel() # todo: test with other base models too

    batch = random_batch(unknown_proportion=1.00)
    loss_DANN(model, batch, 7, 130)

    batch = random_batch(unknown_proportion=0.5)
    loss_DANN(model, batch, 7, 130)

    batch = random_batch(unknown_proportion=0.0)
    loss_DANN(model, batch, 7, 130)
