from typing import Protocol


class MoneyPrinterTurboAdapter(Protocol):
    """
    Boundary around reusable MoneyPrinterTurbo functionality.

    The rest of VideoGen must not import MPT internals directly.
    """

    async def create_video(self, *args, **kwargs): ...
