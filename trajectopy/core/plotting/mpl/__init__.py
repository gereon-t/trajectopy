import logging
import os

import matplotlib.pyplot as plt

logger = logging.getLogger("root")

plt.rcParams["figure.max_open_warning"] = 50


def mplstyle_file_path() -> str:
    custom_path = os.path.join("./custom.mplstyle")
    if os.path.isfile(custom_path):
        logger.info("Using custom matplotlib style from %s", custom_path)
        return custom_path

    logger.info(
        "Using default settings for matplotlib style. You can use custom styles by creating a 'custom.mplstyle' file in the current directory."
    )
    return os.path.join(os.path.dirname(__file__), "default.mplstyle")


base_path = os.path.join(os.path.dirname(__file__))
MPL_STYLE_PATH = mplstyle_file_path()
plt.style.use(MPL_STYLE_PATH)
logger.info("Using matplotlib style: %s", MPL_STYLE_PATH)
