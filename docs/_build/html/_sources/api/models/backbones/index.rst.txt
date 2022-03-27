Backbones
==========

.. toctree::
    ast <ast>
    timesformer <timesformer>
    mm_transformer <mm_transformer>


Dividing images to sequence of pathches
----------------------------------------

PatchEmbed
~~~~~~~~~~~

Extraction of patches from series of images (video)


.. autoclass:: liedet.models.backbones.timesformer.PatchEmbed
    :members:
    :special-members:
    :show-inheritance:


Extraction of patches from audio spectrogram


.. autoclass:: liedet.models.backbones.ast.PatchEmbed
    :members:
    :special-members:
    :show-inheritance:


Transformer
------------

TransformerEncoder
~~~~~~~~~~~~~~~~~~~

.. autoclass:: liedet.models.common.transformer.TransformerEncoder
    :members:
    :special-members:
    :exclude-members: __init__
    :show-inheritance:


Attention Layers
-----------------

DividedTemporalAttentionWithNorm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: liedet.models.common.transformer.DividedTemporalAttentionWithNorm
    :members:
    :show-inheritance:
    :special-members:


DividedSpatialAttentionWithNorm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: liedet.models.common.transformer.DividedTemporalAttentionWithNorm
    :members:
    :show-inheritance:
    :special-members:


Feed-Forward
-------------

FFNWithNorm
~~~~~~~~~~~~

.. autoclass:: liedet.models.common.transformer.FFNWithNorm
    :members:
    :show-inheritance:
    :special-members:


Multimodal Transformer
-----------------------

.. autoclass:: liedet.models.backbones.mm_transformer.AttentionBottleneckTransformer
    :members:
    :show-inheritance:
    :special-members:
