import json
import numpy as np

from holowizard.livereco.client import module_context
from holowizard.core.parameters.type_conversion.json_writable import JsonWritable


def send(function_name, **kwargs):
    dictionary = {"function": function_name}

    for k, v in kwargs.items():
        if not isinstance(v, str) and not isinstance(v, np.ndarray):
            dictionary[k] = v.to_json()
        elif isinstance(v, np.ndarray):
            dictionary[k] = json.dumps(JsonWritable.get_array(v), default=lambda o: o.__dict__)
        else:
            dictionary[k] = v

    if not module_context.network_socket:
        raise RuntimeError("Connection to liverco server not initialized")

    module_context.network_socket.send_json(dictionary)
