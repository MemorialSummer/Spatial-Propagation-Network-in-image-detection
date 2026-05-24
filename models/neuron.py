class Neuron:
    def __init__(self, nid, x, y, z):
        self.nid = nid

        self.x = x
        self.y = y
        self.z = z

        self.value = 0.0
        self.bias = 0.0

        self.in_connections = []
        self.out_connections = []