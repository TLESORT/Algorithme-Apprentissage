from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer

import numpy as np
from random import randint

class randomPixelKill(SourcewiseTransformer, ExpectsAxisLabels):
    def __init__(self, data_stream, image_shape, seed=1,**kwargs):
        self.image_shape = image_shape


        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(randomPixelKill, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
#        print("a batch transform is starting")
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        if isinstance(source, np.ndarray) and source.ndim == 4:
            return [self.transform_source_example(im, source_name)
                    for im in source]
             
        elif all([isinstance(b, np.ndarray) and b.ndim == 3 for b in source]):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)

        if not isinstance(example, np.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
	rand =np.random.binomial(1,0.01, np.shape(example))
	#modified=(example and not(rand)) or (not(example) and rand)
	modified = example*(np.logical_xor(example,rand)).astype(np.uint8)
	modified = modified+rand*np.random.random_integers(256,size=np.shape(example))

        return modified.astype(np.uint8)
