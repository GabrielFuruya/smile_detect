class Config(object):
    """
    Config Class
    """
    DEBUG = False
    TESTING = False

class Development(Config):
    """
    Development Class
    """
    DEBUG = True

class Production(Config):
    """
    Production Class
    """
    DEBUG = False
