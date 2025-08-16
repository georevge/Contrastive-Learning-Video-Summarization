# -*- coding: utf-8 -*-
from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    """ Main function that sets the data loaders; trains and evaluates the model."""
    config = get_config(mode='train')
    val_config = get_config(mode='validation')

    print(config)
    print(val_config)
    print('Currently selected split_index:', config.split_index)
    train_loader = get_loader(config.mode, config.video_type, config.split_index)
    validation_loader = get_loader(val_config.mode, val_config.video_type, val_config.split_index)
    solver = Solver(config, train_loader, validation_loader)

    solver.build()
    solver.evaluate(-1)	 # evaluates the summaries using the initial random weights of the network
    solver.train()
# tensorboard --logdir '../PGL-SUM/Summaries/PGL-SUM/'
