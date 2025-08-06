import traceback
import zmq
import holowizard.livereco
from holowizard.livereco.client import module_context
from holowizard.livereco.client.send import send


def connect(ip, port=holowizard.livereco.server_port):
    address = "tcp://" + ip + ":" + str(port)
    try:
        if module_context.network_socket:
            module_context.network_socket.close(linger=0)
        if module_context.network_context:
            module_context.network_context.destroy(linger=0)
        module_context.network_context = zmq.Context()
        module_context.network_socket = module_context.network_context.socket(zmq.PUSH)
        module_context.network_socket.connect(address)

        send("ping")

        return True

    except Exception:
        traceback.print_exc()
        return False
