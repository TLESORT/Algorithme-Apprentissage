
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream, ServerDataStream
from fuel.schemes import ShuffledScheme
#from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, MaximumImageDimensions, Random2DRotation
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, Random2DRotation
from fuel.transformers import Flatten, Cast, ScaleAndShift
from fuel.server import start_server
from argparse import ArgumentParser
from ScikitResize import ScikitResize
from Flipped import FlipAsYouCan

#This server have to be run for valid and train at the same time to make the CatVsDog works with : 
# python data_server.py 
#and
# python data_server.py --type valid
#PARAMETERS
image_size = (128,128) #should be the same than the image size of the model
batch_size = 100

#Code legerement modifie de Florian Bordes

#Function to get and process the data
def create_data(data, size, batch_size,_port):
        if data == "train":
                cats = DogsVsCats(('train',), subset=slice(0, 20000))
                port = _port+2
        elif data == "valid":
                cats = DogsVsCats(('train',), subset=slice(20000, 25000))
                port = _port+3
        print 'port', port
        stream = DataStream.default_stream(cats, iteration_scheme=ShuffledScheme(cats.num_examples, batch_size))
        stream_downscale = MinimumImageDimensions(stream, size, which_sources=('image_features',))
	stream_rotate = FlipAsYouCan(stream_downscale,)
        stream_max = ScikitResize(stream_rotate, image_size, which_sources=('image_features',))
        stream_scale = ScaleAndShift(stream_max, 1./255, 0, which_sources=('image_features',))
        stream_data = Cast(stream_scale, dtype='float32', which_sources=('image_features',))
        start_server(stream_data, port=port)

if __name__ == "__main__":
        parser = ArgumentParser("Run a fuel data stream server.")
        parser.add_argument("--type", type=str, default="train",
                help="Type of the dataset (train, valid)")
        parser.add_argument("--port",type=int,default=5560)
        args = parser.parse_args()
        if args.type=="valid" or args.type=="train":
                create_data(args.type, image_size, batch_size,args.port)
        else:
                print "wrong input"

