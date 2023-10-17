#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""General utils.

.. currentmodule:: health_multimodal.common

.. autosummary::
   :toctree:

   device
   visualization
"""

from .visualization import plot_phrase_grounding_similarity_map

__all__ = [plot_phrase_grounding_similarity_map]
