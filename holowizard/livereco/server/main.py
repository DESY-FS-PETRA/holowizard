#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 13:35:22 2021

@author: joh
"""

import zmq
import traceback
import holowizard.livereco_server
from holowizard.livereco_server.server.flatfield_correction import FlatfieldCorrection
from holowizard.livereco_server.server.reconstruction import Reconstruction
from holowizard.livereco_server.server.find_focus import FindFocus
from holowizard.core.logging.logger import Logger
from holowizard.core.api.viewer import ZeroMQViewer
from holowizard.core.api.viewer import LossViewer


def main():
    status_port = 8557
    status_context = zmq.Context()
    status_socket = status_context.socket(zmq.PUSH)
    status_socket.bind("tcp://*:" + str(status_port))

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:" + str(holowizard.livereco_server.server_port))

    zeromqViewer = ZeroMQViewer(holowizard.livereco_server.viewer_port)
    lossViewer = LossViewer()
    flat = FlatfieldCorrection([zeromqViewer, lossViewer])
    rec = Reconstruction([zeromqViewer, lossViewer])
    findfoc = FindFocus([zeromqViewer, lossViewer])

    def send(function_name, **kwargs):
        dictionary = {"function": function_name}

        for k, v in kwargs.items():
            dictionary[k] = v

        if not status_socket:
            raise RuntimeError("Status Socket initialized")
        status_socket.send_json(dictionary)

    while True:
        Logger.current_log_level = Logger.level_num_image_info

        print("Waiting for calls...")
        message = socket.recv_json()
        print("Calling: ", message["function"])

        try:
            function = message["function"]

            if function == "ping":
                send(function_name="pong")
                continue

            if function == "reconfigure_logger":
                Logger.configure(
                    working_dir=message["working_dir"],
                    session_name=message["session_name"],
                )
                continue

            if function == "reconstruct":
                rec.reconstruct_x(message["flatfield_correction_params"], message["reco_params"])
                continue

            if function == "find_focus":
                current_log_level = Logger.current_log_level
                Logger.current_log_level = Logger.level_num_image_final
                found_z01 = findfoc.find_focus(message["flatfield_correction_params"], message["reco_params"])
                Logger.current_log_level = current_log_level
                send(function, found_z01=found_z01)

                continue

            if function == "correct_flatfield":
                flat.correct_flatfield(message["flatfield_correction_params"])
                continue

            if function == "calculate_flatfield_components":
                flat.calc_flatfield_components(message["flatfield_components_params"])
                continue

            if function == "add_flatfield":
                flat.add_flatfield(message["measurement"])

            if function == "reset_flatfield_list":
                flat.reset_flatfield_list()

        except:
            traceback.print_exc()


if __name__ == "__main__":
    main()
