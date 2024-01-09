#!./env python

"""
    deprecated
"""


class Env:
    def __init__(self, device='cpu'):
        self.device = device

env = Env()

def set_env(device='cpu'):
    global env
    env = Env(device=device)
