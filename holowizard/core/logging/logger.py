import os
import logging
import numpy
from skimage import io
import datetime
import time
import pickle
import io as io_low_level
from .header import Header
from . import *


class Logger:
    # logging.INFO = 20, higher number means less output, higher severity of error.
    # logging DEBUG = 10
    level_num_image_debug = logging.DEBUG - 1
    level_num_image_info = logging.INFO - 5
    level_num_image_final = logging.INFO - 4
    level_num_loss = logging.INFO - 3
    level_num_params = logging.INFO - 2
    level_num_comment = 99
    level_num_header = 100

    current_log_level = level_num_loss
    current_logger = None

    initialized = False
    working_dir = None
    session_name = None

    @staticmethod
    def get_timestamp():
        timestamp = time.time()
        st = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
        return st

    @staticmethod
    def init():
        if Logger.initialized == True:
            return

        Logger.add_log_level("DEBUG", logging.DEBUG, Logger.debug_level_callback)
        Logger.add_log_level("INFO", logging.INFO, Logger.info_level_callback)
        Logger.add_log_level("HEADER", Logger.level_num_header, Logger.header_level_callback)
        Logger.add_log_level("COMMENT", Logger.level_num_comment, Logger.comment_level_callback)
        Logger.add_log_level("LOSS", Logger.level_num_loss, Logger.loss_level_callback)
        Logger.add_log_level(
            "IMAGE_DEBUG",
            Logger.level_num_image_debug,
            Logger.image_debug_level_callback,
        )
        Logger.add_log_level("IMAGE_INFO", Logger.level_num_image_info, Logger.image_info_level_callback)
        Logger.add_log_level(
            "IMAGE_FINAL",
            Logger.level_num_image_final,
            Logger.image_final_level_callback,
        )
        Logger.add_log_level("PARAMS", Logger.level_num_params, Logger.params_level_callback)

        Logger.current_logger = logging.getLogger("root")

        Logger.initialized = True

    @staticmethod
    def configure(working_dir=None, session_name=None):
        if Logger.initialized == False:
            Logger.init()

        if working_dir != None:
            Logger.working_dir = working_dir

        if session_name != None:
            Logger.session_name = session_name

        logger = logging.getLogger(session_name)
        logger.setLevel(Logger.current_log_level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(Logger.current_log_level)
        console_handler.setFormatter(
            # logging.Formatter("%(asctime)s - %(name)s - %(levelname)11s - %(message)s")
            logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        )

        logger.addHandler(console_handler)

        if Logger.working_dir is not None:
            try:
                output_path = Logger.working_dir + "/" + Logger.session_name
                os.makedirs(output_path, exist_ok=True)
                log_file = output_path + "/" + Logger.get_timestamp() + "_log.txt"

                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(Logger.current_log_level)
                file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
                logger.addHandler(file_handler)
            except OSError:
                print("Error creating log file " + output_path)
                pass

        Logger.current_logger = logger

        Logger.log_header()

    @staticmethod
    def log_header():
        messages = Header.get_header(Logger.session_name, Logger.working_dir)

        for message in messages:
            logging.header(message)

    @staticmethod
    def add_log_level(level_name, level_num, log_callback):
        logging.addLevelName(level_num, level_name)

        def log_for_level(self, message, *args, **kwargs):
            if self.isEnabledFor(level_num):
                self._log(level_num, message, args, **kwargs)

        setattr(logging, level_name, level_num)
        setattr(logging.getLoggerClass(), level_name.lower(), log_for_level)
        setattr(logging, level_name.lower(), log_callback)

    @staticmethod
    def debug_level_callback(message, *args, **kwargs):
        if not Logger.current_logger.isEnabledFor(logging.DEBUG):
            return
        Logger.current_logger.log(logging.DEBUG, message, *args, **kwargs)

    @staticmethod
    def info_level_callback(message, *args, **kwargs):
        if not Logger.current_logger.isEnabledFor(logging.INFO):
            return
        Logger.current_logger.log(logging.INFO, message, *args, **kwargs)

    @staticmethod
    def header_level_callback(message, *args, **kwargs):
        if not Logger.current_logger.isEnabledFor(Logger.level_num_header):
            return
        Logger.current_logger.log(Logger.level_num_header, message)

    @staticmethod
    def comment_level_callback(message, *args, **kwargs):
        if not Logger.current_logger.isEnabledFor(Logger.level_num_comment):
            return
        comment_space = "    "
        message = comment_space + str(message).strip() + comment_space
        block_border = "{s:{c}^{n}}".format(s="", n=comment_block_length, c=comment_character)
        message_to_log = "{s:{c}^{n}}".format(s=message, n=comment_block_length, c=comment_character)
        Logger.current_logger.log(Logger.level_num_comment, block_border)
        Logger.current_logger.log(Logger.level_num_comment, message_to_log)
        Logger.current_logger.log(Logger.level_num_comment, block_border)

    @staticmethod
    def loss_level_callback(message):
        Logger.current_logger.log(Logger.level_num_loss, message)

    @staticmethod
    def image_debug_level_callback(data_name, data):
        if not Logger.current_logger.isEnabledFor(Logger.level_num_image_debug):
            return
        Logger.image_level_callback(data_name, data, Logger.level_num_image_debug)

    @staticmethod
    def image_info_level_callback(data_name, data):
        if not Logger.current_logger.isEnabledFor(Logger.level_num_image_info):
            return
        Logger.image_level_callback(data_name, data, Logger.level_num_image_info)

    @staticmethod
    def image_final_level_callback(data_name, data):
        if not Logger.current_logger.isEnabledFor(Logger.level_num_image_final):
            return
        Logger.image_level_callback(data_name, data, Logger.level_num_image_final)

    @staticmethod
    def image_level_callback(data_name, data, level):
        data_path = Logger.working_dir + "/" + Logger.session_name + "/" + Logger.get_timestamp() + "_" + data_name

        data_path_tiff = data_path + ".tiff"
        Logger.current_logger.log(level, data_path_tiff)
        io.imsave(data_path_tiff, data)

    @staticmethod
    def params_level_callback(data_name, data):
        if not Logger.current_logger.isEnabledFor(Logger.level_num_params):
            return

        data_path = (
            Logger.working_dir + "/" + Logger.session_name + "/" + Logger.get_timestamp() + "_" + data_name + ".pkl"
        )

        Logger.current_logger.log(Logger.level_num_params, data_path)

        f = io_low_level.BytesIO()
        pickle.dump(data, f)
        serialized = f.getvalue()
        with open(data_path, "wb") as file:
            file.write(serialized)

    @staticmethod
    def custom_string_file(keyword, string_list):
        file_name = (
            Logger.working_dir + "/" + Logger.session_name + "/" + Logger.get_timestamp() + "_" + keyword + ".txt"
        )
        with open(file_name, "w") as f:
            for string in string_list:
                f.write(string + "\n")
