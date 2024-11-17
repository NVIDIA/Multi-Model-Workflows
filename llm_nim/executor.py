# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

import importlib
import os
import sys
import tempfile
import traceback


class Executor:

    def __init__(self, function_name="postprocessor"):
        """Constructor to setup the code executor."""
        self.postprocessor = None
        self.function_name = function_name

    def save_python_function(self, function_string):
        """Execute a function based on output from CodeLLAMA."""
        _, temp_pyfile = tempfile.mkstemp(suffix=".py")
        with open(temp_pyfile, "w") as codefile:
            codefile.write(function_string)
        return temp_pyfile
    
    def load_function_from_string(self, function_string):
        """Loaded function value."""
        pyfile = self.save_python_function(function_string=function_string)
        sys.path.append(os.path.dirname(pyfile))
        try:
            module = importlib.import_module(
                os.path.basename(pyfile)[:-3]
            )
            self.postprocessor = getattr(module, self.function_name)
        except Exception as e:
            traceback_str = traceback.format_exc()
    
    def execute(self, metadata):
        """Execute the function."""
        return self.postprocessor(metadata)

    def __del__(self):
        """Pop path from system paths."""
        if "/tmp" in sys.path:
            sys.path.pop(sys.path.index("/tmp"))
    
