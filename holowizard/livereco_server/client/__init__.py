from dataclasses import dataclass
import zmq


@dataclass
class ModuleContext:
    network_context: zmq.Context = None
    network_socket: zmq.Socket = None


module_context = ModuleContext()
