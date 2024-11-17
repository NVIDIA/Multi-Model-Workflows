from dataclasses import dataclass
import inspect
import json
import logging
import os
import subprocess
import sys

from utils.constants import APP_CACHE
from utils.utils import execute_command

logger = logging.getLogger(__name__)


class ModelInstance:
    """A model instance to be registered for primary inference."""

    def __init__(self,
                 name: str = None,
                 version: str = None,
                 entrypoint: str = None,
                 team: str = None,
                 org: str = None,
                 local_model_path: str = None):
        
        """Constructor."""
        self.model_name = name
        self.model_version = version
        self.entrypoint = entrypoint
        self.team = team
        self.org = org
        self.ngc_command_prefix = "ngc registry model"
        self.default_format = "--format_type json"

        self.local_model_path = local_model_path
        assert self._check_ngc_binary(), "NGC binary doesn't exist."

    @staticmethod
    def _check_ngc_binary():
        """Check if a ngc binary exists."""
        command_help = "ngc --help"
        return execute_command(command_help, stdout=subprocess.DEVNULL)

    @property
    def ngc_model_url(self):
        """Get the ngc model url."""
        assert all([
            self.team is not None,
            self.org is not None,
            self.model_name is not None,
            self.model_version is not None,
        ]), "Model needs to be instantiated with valid particular."
        return f"{self.org}/{self.team}/{self.model_name}:{self.model_version}"
    
    @property
    def local_path(self):
        """Default local path."""
        if self.local_model_path:
            return self.local_model_path
        
        app_cache = APP_CACHE
        model_cache = os.path.dirname(app_cache)
        os.makedirs(model_cache, exist_ok=True)
        model_path = os.path.join(model_cache, f"{self.model_name}_v{self.model_version}")
        return model_path

    def retrieve_model_metadata(self):
        """Get metadata for the model from NGC."""
        inspect_command = f"{self.ngc_command_prefix} info {self.ngc_model_url}"
        print(f"Running command: {inspect_command}")
        return self.execute_command_parse_output(inspect_command) 
    
    def execute_command_parse_output(self, command):
        """Execute an NGC command and parse output."""
        run_command = f"{command} {self.default_format}"
        data = subprocess.check_output(
            run_command,
            executable="/bin/bash",
            shell=True,
            env=os.environ
        )
        return json.loads(data)
    
    def download_model(self):
        """Download the model to the local path."""
        download_prefix = f"{self.ngc_command_prefix} download-version {self.ngc_model_url}"
        if not all([
            os.path.exists(self.local_path),
            os.path.isdir(self.local_path)]):
            destination_path = os.path.dirname(self.local_path)
            download_command = f"{download_prefix} --dest {destination_path}"
            data = self.execute_command_parse_output(download_command)
        else:
            assert os.listdir(self.local_path), "There are no models in the cache directory."
            print(f"The model was already downloaded and available in {self.local_path}")
            [print(file) for file in os.listdir(self.local_path)]
        return self.local_path

    def __str__(self):
        """Return model instance data."""
        inspectable_members = [item for item in inspect.getmembers(self) if not item[0].startswith("_")]
        string_representation = ""
        for member in inspectable_members:
            if not inspect.ismethod(member[1]):
                string_representation += f"{member[0]}: {member[1]}\n"
        return string_representation
