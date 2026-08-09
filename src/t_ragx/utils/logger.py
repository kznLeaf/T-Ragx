import logging
import sys

from colorama import Back, Fore, Style


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(
        self, *args, colors: dict[str, str] | None = None, **kwargs
    ) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL

        return super().format(record)


formatter = ColoredFormatter(
    "{asctime} {color}[{levelname:.1s}] {message}{reset}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    colors={
        "DEBUG": Fore.CYAN + Style.BRIGHT,
        "INFO": Fore.GREEN + Style.BRIGHT,
        "WARNING": Fore.YELLOW + Style.BRIGHT,
        "ERROR": Fore.RED + Style.BRIGHT,
        "CRITICAL": Fore.RED + Style.BRIGHT + Back.YELLOW,
    },
)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logging.basicConfig()
logger = logging.getLogger("t_ragx")

logger.handlers[:] = []
logger.addHandler(handler)
