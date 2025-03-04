class SynapseGen:
    """
    This class will represents a single synapse (link) gene for the NEAT algorithm
    Atributes:
        1) Synapse link
        2) Double weight
        3) Boolean is_enable
    Methods:
        1) Pending
    """
    def __init__(self,synapse,w):
        self.link = synapse
        self.weight = w
        self.is_enable = True