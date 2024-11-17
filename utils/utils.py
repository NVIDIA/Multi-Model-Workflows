import logging
import os
import subprocess
import sys

from typing import List

logger = logging.getLogger(__name__)


def execute_command(command: str|List[str],
                    shell: str="/bin/bash",
                    env=os.environ,
                    stdout=sys.stdout):
    """Simple wrapper to execute a subprocess."""
    try:
        subprocess.run(
            command,
            shell=True,
            env=env,
            executable=shell,
            stdout=stdout,
            check=True
        )
        return True
    except subprocess.CalledProcessError as exc:
        logger.info("Couldn't execute command: {command}.".format(command=command))
        if hasattr(exc, "output"):
            if exc.output is not None:
                logger.info(exc.output)
        return False


def check_and_create(path):
    """Check and create the path."""
    os.makedirs(path, exist_ok=True)
    return path
