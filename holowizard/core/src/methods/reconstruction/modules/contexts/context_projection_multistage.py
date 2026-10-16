import torch
from dataclasses import dataclass
from typing import List

from holowizard.core.src.methods.reconstruction.modules.contexts.context_projection_singlestage import (
    ContextProjectionSingleStage,
)
from holowizard.core.src.methods.reconstruction.modules.contexts.abstract_contexts.abstract_context_multistage import (
    AbstractContextMultistage,
)


@dataclass(init=True)
class ContextProjectionMultistage(AbstractContextMultistage):
    current_projection_singlestage: ContextProjectionSingleStage
