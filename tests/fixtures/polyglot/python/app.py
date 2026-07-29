from lib import Engine, helper


def process(x):
    return helper(x)


def run():
    return Engine().process(process(1))
