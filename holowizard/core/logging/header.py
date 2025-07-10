import git
import time
import datetime
import getpass
import sys
from enum import Enum

from . import *


class Header:
    class HeaderCommands(Enum):
        LINE = 0
        TEXT = 1
        LOGO = 2

    def __init__(self):
        None

    @staticmethod
    def get_header(session_name, working_directory):
        header_message = []
        try:
            header_message.append(
                " _   _         _         _    _  _                      _ "
            )
            header_message.append(
                "| | | |       | |       | |  | |(_)                    | |"
            )
            header_message.append(
                "| |_| |  ___  | |  ___  | |  | | _  ____ __ _  _ __  __| |"
            )
            header_message.append(
                "|  _  | / _ \ | | / _ \ | |/\| || ||_  // _` || '__|/ _` |"
            )
            header_message.append(
                "| | | || (_) || || (_) |\  /\  /| | / /| (_| || |  | (_| |"
            )
            header_message.append(
                "\_| |_/ \___/ |_| \___/  \/  \/ |_|/___|\__,_||_|   \__,_|"
            )
            header_message.append(
                "                                                          "
            )

            header_message.append(Header.get_line())
            header_message.append(Header.get_text("Session Name:   " + session_name))
            header_message.append(
                Header.get_text("Logs Directory: " + working_directory)
            )
            header_message.append(Header.get_line())

            header_message.append(
                Header.get_text("Local User:     " + Header.user_name())
            )
            header_message.append(
                Header.get_text("Python Path     " + Header.python_interpreter())
            )
            header_message.append(Header.get_line())
        except:
            None

        try:
            header_message.append(
                Header.get_text("Git Repo:       " + Header.git_repo_name())
            )
            header_message.append(
                Header.get_text("Git Branch:     " + Header.git_branch())
            )
            header_message.append(
                Header.get_text("Git Revision:   " + Header.git_revision())
            )
        except:
            None

        header_message.append(Header.get_line())

        header_message.append(
            "                                                                          "
        )
        return header_message

    @staticmethod
    def get_line():
        return "{s:{c}^{n}}".format(s="", n=comment_block_length, c=comment_character)

    @staticmethod
    def get_text(message):
        message_to_log = "{s:{c}^{n}}".format(
            s="", n=comment_alignment_length, c=comment_character
        )
        message_to_log = message_to_log + " " + str(message)
        message_to_log = "{s:{c}<{n}}".format(
            s=message_to_log, n=comment_block_length, c=" "
        )
        return message_to_log

    @staticmethod
    def time():
        time_stamp = time.time()
        time_formatted = datetime.datetime.fromtimestamp(time_stamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return time_formatted

    @staticmethod
    def user_name():
        return str(getpass.getuser())

    @staticmethod
    def python_interpreter():
        return str(sys.executable)

    @staticmethod
    def git_repo_name():
        repo = git.Repo(search_parent_directories=True)
        name = repo.remotes.origin.url
        return str(name)

    @staticmethod
    def git_revision():
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return str(sha)

    @staticmethod
    def git_branch():
        repo = git.Repo(search_parent_directories=True)
        branch = repo.active_branch
        return str(branch)
