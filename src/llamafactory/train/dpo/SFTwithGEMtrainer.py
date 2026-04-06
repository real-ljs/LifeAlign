# trainers/SFTwithGEMTrainer.py

from .SFTtrainer import SFTTrainer
from ...extras.logging import get_logger

logger = get_logger(__name__)

class SFTwithGEMTrainer(SFTTrainer):
    """
    一个用于GEM方法的SFT Trainer占位符。
    所有GEM的核心逻辑都由外部的GEMManager和共享的TaskGradientCallback处理。
    这个类本身不需要额外的逻辑。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initialized SFTwithGEMTrainer.")