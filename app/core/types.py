from enum import Enum

class CombineRule(str, Enum):
    ADD = "add"
    MUL = "mul"
    LOG = "log"