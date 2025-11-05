import threading
import traceback
import zmq
import holowizard.livereco
from holowizard.livereco.client import controller_context
from holowizard.livereco.client import status_context
from holowizard.livereco.client.send import send

def stop_status_polling():
    status_context.status_poll_thread_stop = True

def disconnect():
    stop_status_polling()
    status_context.status_poll_thread.join()

def connect_controller(address, port=holowizard.livereco.server_port):
    tcp_address = "tcp://" + str(address) + ":" + str(port)

    try:
        if controller_context.network_socket:
            controller_context.network_socket.close(linger=0)
        if controller_context.network_context:
            controller_context.network_context.destroy(linger=0)
        controller_context.network_context = zmq.Context()
        controller_context.network_socket = controller_context.network_context.socket(zmq.PUSH)
        controller_context.network_socket.connect(tcp_address)

        send("ping")

        return True

    except Exception:
        traceback.print_exc()
        return False


def poll_status():
    status_poller = zmq.Poller()
    status_poller.register(status_context.network_socket, flags=zmq.POLLIN)
    while not status_context.status_poll_thread_stop:
        socks = dict(status_poller.poll(1000))
        status = None
        if (
                status_context.network_socket in socks
                and socks[status_context.network_socket] == zmq.POLLIN
        ):
            status = status_context.network_socket.recv_json()
        if status and status["function"] == "find_focus":
            print(status["found_z01"])


def connect_status_poller(address,port=holowizard.livereco.status_port):
    status_context.network_context = zmq.Context()
    status_context.status_poller = zmq.Poller()

    status_context.network_socket = status_context.network_context.socket(zmq.PULL)
    tcp_address = "tcp://" + str(address) + ":" + str(port)
    print("Connecting to", tcp_address)
    status_context.network_socket.connect(tcp_address)
    print("Socket created")

    status = None
    if status_context.status_poll_thread and status_context.status_poll_thread.is_alive():
        status_context.status_poll_thread_stop = True
        status_context.status_poll_thread.join()
        status_context.status_poll_thread_stop = False

    print("Trying to connect")
    if status_context.network_socket and connect_controller(address):
        print("Register poller")
        status_context.status_poller.register(status_context.network_socket, zmq.POLLIN)
        print("Polling status")
        socks = dict(status_context.status_poller.poll(5000))
        if (
                status_context.network_socket in socks
                and socks[status_context.network_socket] == zmq.POLLIN
        ):
            status = status_context.network_socket.recv_json()

    if status and status["function"] == "pong":
        status_context.status_poll_thread = threading.Thread(target=poll_status)
        status_context.status_poll_thread.start()
        return True

    return False
