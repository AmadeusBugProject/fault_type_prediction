import os
import pathlib


def root_dir():
    return str(pathlib.Path(__file__).parent) + '/'
