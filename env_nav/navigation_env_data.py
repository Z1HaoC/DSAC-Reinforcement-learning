from navigation_env import GazeboEnv


def env_creator(**kwargs):
    env = GazeboEnv(**kwargs)
    return  env
