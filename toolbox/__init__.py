# flake8: noqa

import pkg_resources


# __version__ = pkg_resources.get_distribution('chainer_mask_rcnn').version
__version__ = "0.0.1"


from . import datasets
from . import extensions
from . import functions
from . import models
from . import utils
