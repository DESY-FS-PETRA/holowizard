import traceback
import zmq
import holowizard.livereco
from holowizard.livereco.client import controller_context
from holowizard.livereco.client.send import send


def connect(ip, port=holowizard.livereco.server_port):
    address = "tcp://" + ip + ":" + str(port)
    try:
        if controller_context.network_socket:
            controller_context.network_socket.close(linger=0)
        if controller_context.network_context:
            controller_context.network_context.destroy(linger=0)
        controller_context.network_context = zmq.Context()
        controller_context.network_socket = controller_context.network_context.socket(zmq.PUSH)
        controller_context.network_socket.connect(address)

        send("ping")

        return True

    except Exception:
        traceback.print_exc()
        return False

