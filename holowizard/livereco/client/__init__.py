import threading
from dataclasses import dataclass
import zmq


@dataclass
class BasicContext:
    network_context: zmq.Context = None
    network_socket: zmq.Socket = None


@dataclass
class StatusContext(BasicContext):
    status_poller: zmq.Poller = None
    status_poll_thread: threading.Thread = None
    status_poll_thread_stop: bool = False


controller_context = BasicContext()
status_context = StatusContext()
