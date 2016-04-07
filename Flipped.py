from __future__ import division
from io import BytesIO
import math

import numpy
from fuel.transformers import ExpectsAxisLabels, Transformer, SourcewiseTransformer,AgnosticSourcewiseTransformer
from PIL import Image
from six import PY3

try:
    from fuel.transformers.image._image import window_batch_bchw
    window_batch_bchw_available = True
except ImportError:
    window_batch_bchw_available = False
from fuel.transformers.image import ExpectsAxisLabels, SourcewiseTransformer
from fuel.transformers import config

class FlipAsYouCan(Transformer, ExpectsAxisLabels):
    """Randomly rotate 2D images in the spatial plane.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    maximum_rotation : float, default `math.pi`
        Maximum amount of rotation in radians. The image will be rotated by
        an angle in the range [-maximum_rotation, maximum_rotation].
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.
    Notes
    -----
    This transformer expects to act on stream sources which provide one of
     * Single images represented as 3-dimensional ndarrays, with layout
       `(channel, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, channel, height, width)`.
    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.
    """ 
    def __init__(self, data_stream,resample='nearest', **kwargs):
        self.maximum_rotation = 180
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))

        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
	data_stream.produces_examples=False
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(FlipAsYouCan, self).__init__(data_stream,**kwargs)

      
    def transform_batch(self, batch):	
	output = ([])
	output.append([])
	angles=([])
	Flipped = numpy.random.binomial(1,0.5, len(batch[0]) )
        if isinstance(batch[0], list) and all(isinstance(b, numpy.ndarray) and b.ndim == 3 for b in batch[0]):
	    #c'est ici que passe le programme pour le dataset chien chat
	    for im, angle in zip(batch[0], Flipped):
            	output[0].append(self._example_transform(im, angle*180))
		angles.append([angle])
	else:
	    for im, angle in zip(batch[0], Flipped):
            	output[0].extend(numpy.array([self._example_transform(im, angle)],dtype=batch[0].dtype))
		
	output.append(numpy.array(angles,dtype=batch[1].dtype))
        
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example, rotation_angle):
	if rotation_angle==180:
		dt = example.dtype
		im = Image.fromarray(example.transpose(1, 2, 0))
		example = numpy.array(im.rotate(rotation_angle,resample=self.resample)).astype(dt)
		example=example.transpose(2, 0, 1)
        return example



