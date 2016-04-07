"""convolutional network example.
The original version of this code come from : https://github.com/mila-udem/blocks-examples/blob/master/mnist_lenet/
This code have been developed with the help of  : 
Florian Bordes (the code of the data server come from him)
Thomas Geoges (the code of the bokeh plot come from him)
Vincent Antaki (the code of the ScikitResize come from him)


this Code need to be run with 3 others : 
1 data server for training image
1 data erver for valid images
1 server for online printing

more information about it on : 
https://fuel.readthedocs.org/en/latest/server.html
https://blocks.readthedocs.org/en/latest/plotting.html
"""
import logging
import numpy
from argparse import ArgumentParser

from theano import tensor

from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,Softmax, Activation)
from blocks.algorithms import GradientDescent, Scale, Adam, Momentum
from blocks.initialization import Constant, Uniform

from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation

from blocks_extras.extensions.plot import Plot
from fuel.streams import ServerDataStream
import socket
import datetime
from argparse import ArgumentParser

from ScikitResize import ScikitResize
from LeNet import LeNet
import random

def main(port_data,LR,Init1,Init2):
    mlp_hiddens = [500]
    filter_sizes = [(3,3),(3,3),(3,3)]
    feature_maps = [20,50,80]
    pooling_sizes = [(2,2),(2,2),(2,2)]
    save_to="DvC_parcour.pkl"
    image_size = (128, 128)
    output_size = 2
    learningRate=LR
    num_epochs=10
    num_batches=None
    if socket.gethostname()=='tim-X550JX':host_plot = 'http://localhost:5011'
    else:host_plot = 'http://hades.calculquebec.ca:5010'

    # Use ReLUs everywhere and softmax for the final prediction
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    convnet = LeNet(conv_activations, 3, image_size,
                    filter_sizes=filter_sizes,
                    feature_maps=feature_maps,
                    pooling_sizes=pooling_sizes,
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='full',
                    weights_init=Uniform(width=Init1),
                    biases_init=Constant(0))

    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    convnet.push_initialization_config()
    convnet.layers[0].weights_init = Uniform(width=Init1)
    convnet.layers[1].weights_init = Uniform(width=Init2)
    convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=Init1)
    convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=Init2)
    convnet.initialize()
    logging.info("Input dim: {} {} {}".format(*convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))
    x = tensor.tensor4('image_features')
    y = tensor.lmatrix('targets')

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    cost = (CategoricalCrossEntropy().apply(y.flatten(), probs).copy(name='cost'))
    error_rate = (MisclassificationRate().apply(y.flatten(), probs).copy(name='error_rate'))

    cg = ComputationGraph([cost, error_rate])

    ########### GET THE DATA #####################
    stream_data_train = ServerDataStream(('image_features','targets'), False, port=port_data)
    stream_data_test = ServerDataStream(('image_features','targets'), False, port=port_data+1)

    # Train with simple SGD
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters,step_rule=Momentum(learning_rate=learningRate,momentum=0.9))


    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = []
    extensions.append(Timing())
    extensions.append(FinishAfter(after_n_epochs=num_epochs,after_n_batches=num_batches))
    extensions.append(DataStreamMonitoring([cost, error_rate],stream_data_test,prefix="valid"))
    extensions.append(TrainingDataMonitoring([cost, error_rate,aggregation.mean(algorithm.total_gradient_norm)],prefix="train",after_epoch=True))
    extensions.append(Checkpoint(save_to))
    extensions.append(ProgressBar())
    extensions.append(Printing())

    #Adding a live plot with the bokeh server
    #type http://localhost:5010 in your browser to visualise the result (you need also to install bokeh 0.10.0 and not 0.11.0)
    extensions.append(Plot(' CNN %s %s @ %s' % ('CNN ', datetime.datetime.now(), socket.gethostname()),
                        channels=[['train_error_rate', 'valid_error_rate'],
                         ['train_total_gradient_norm']], after_epoch=True, server_url=host_plot))

    model = Model(cost)

    main_loop = MainLoop(
        algorithm,
        stream_data_train,
        model=model,
        extensions=extensions)

    main_loop.run()


if __name__ == "__main__":
    parser = ArgumentParser("Run a CNN.")
    parser.add_argument("--port_data", type=int, default=5560)
    args = parser.parse_args()
    for i in range(20):
	rand=random.randint(1,10)*0.01
	rand1=random.randint(1,100)*0.01
	rand2=random.randint(1,100)*0.01
    	main(args.port_data,LR=rand,Init1=rand1,Init2=rand2)

