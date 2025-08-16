# -*- coding: utf-8 -*-
from configs import get_config
from test_solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    """ Main function that sets the data loaders; trains and evaluates the model."""
    test_config = get_config(mode='test')

    print(test_config)
    test_loader = get_loader(test_config.mode, test_config.video_type, test_config.split_index)
    solver = Solver(test_config, test_loader)

    solver.evaluate()

# tensorboard --logdir '../PGL-SUM/Summaries/PGL-SUM/'
