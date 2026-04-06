# trainers/SFTwithGEMTrainer.py

from .SFTtrainer import SFTTrainer
from ...extras.logging import get_logger

logger = get_logger(__name__)

class MySFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initialized MySFTtrainer.")