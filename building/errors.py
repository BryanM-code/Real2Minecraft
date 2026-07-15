"""Building-mode error type (own module so stages avoid importing the orchestrator)."""


class BuildingModeError(RuntimeError):
    """A building-pipeline stage failed in a way the user should act on."""

    def __init__(self, stage: str, msg: str):
        super().__init__("[{}] {}".format(stage, msg))
        self.stage = stage
