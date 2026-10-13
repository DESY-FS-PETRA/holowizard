from dataclasses import dataclass
from typing import List

from holowizard.core.src.methods.reconstruction.modules.contexts.abstract_context_distributed import AbstractContextDistributed
from holowizard.core.src.methods.reconstruction.modules.contexts.abstract_contexts.abstract_context_multistage import AbstractContextMultistage
from holowizard.core.src.methods.reconstruction.modules.contexts.context_projection_multistage import ContextProjectionMultistage


@dataclass(init=True)
class ContextProjectionsSet(AbstractContextDistributed, AbstractContextMultistage):
    projection_contexts: List[ContextProjectionMultistage] 
