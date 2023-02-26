#!/usr/bin/env python3

class Vertex:
    def __init__(self, label, r, C):
        # Label
        self.label = label
        # Translation to frame in world frame
        self.r_in0 = r
        # Rotation matrix that expresses world frame vector in pose frame
        self.C_p0 = C
        # List of vertices connected to this vertex
        self.from_list = []
        # List of vertices that this vertex is connected to
        self.to_list = []
        
    def get_from_list(self):
        return [v.label for v in self.from_list]
    
    def get_to_list(self):
        return [v.label for v in self.to_list]
        
class MapVertex(Vertex):
    def __init__(self, label, r):
        # Label
        self.label = label
        # Translation to point in world frame
        self.r_in0 = r
        # List of vertices connected to this vertex
        self.from_list = []
        
class Edge:
    def __init__(self, V1, V2, meas, weight, type : str=None):
        assert isinstance(V1, Vertex), TypeError("V1 not a Vertex")
        assert isinstance(V2, Vertex), TypeError("V2 not a Vertex")
        self.V_from = V1
        self.V_to = V2
        self.meas = {}
        self.weight = {}
        self.update(meas, weight, type)
        
    def update(self, meas, weight, type : str=None):
        if type is None:
            type = "trans"
        if not (type == "trans" or type == "rot"):
            raise TypeError("Edge type not valid")
        if type == "trans" and not meas.shape == (3,1):
            raise ValueError("Measurement is wrong size for translation")
        if "rot" in type and not meas.shape == (3,3):
            raise ValueError("Measurement is wrong size for rotation")
        # Record values
        self.meas[type] = meas
        self.weight[type] = weight
          
    
class MeasGraph:
    """
    Class defining the measurement (factor) graph for a given problem
    """
    def __init__(self, r_p_in0, C_p_0, r_m_in0):
        """ Constructor for measurement graph in a SLAM problem or subproblem.
        Expects ground truth input values.

        Args:
            r_p_in0: (list) of translations to pose frame from world frame in world frame
            C_p_0:   (list) of spatial math rotation matrices rotating base frame vector to pose frame
            r_m_in0: (list) of translations to map point in world frame
        """
        # Pose Vertex Set - dictionary indexed on vertex labels
        self.Vp = dict()
        # Map or Landmark Vertex Set - dictionary indexed on vertex labels
        self.Vm = dict()
        # Edge Structure
        # dict of dicts indexed on vertex sets
        # Stores measurement and weights
        self.E = dict()
        # Make sure that pose length data is consistent
        if not len(C_p_0) == len(r_p_in0):
            raise Exception("Input Pose data has inconsistent length")
        # Loop through poses and define vertices
        for i in range(len(r_p_in0)):
            # Check shape
            assert r_p_in0[i].shape == (3,1), f"Pose translation {i} not (3,1)"
            assert C_p_0[i].shape == (3,3), f"Pose Rotation {i} not (3,3)"
            # Add to list
            label = f'x{i}'
            self.Vp[f'x{i}'] = Vertex(label, r_p_in0[i], C_p_0[i])
        # Loop through map points and define vertices
        for i in range(len(r_m_in0)):
            # Check shape
            assert r_m_in0[i].shape == (3,1),f"Map translation {i} not (3,1)"
            label = f'm{i}'
            self.Vm[label] = MapVertex(label,r_m_in0[i])
    
    def find_edge_verts(self,V_from,V_to):
        # If just strings given, convert to vertex objects
        if type(V_from) == str:
            if "x" in V_from:
                V_from = self.Vp[V_from]
            elif "m" in V_from:
                raise TypeError("Outgoing vertex cannot be Map Vertex")
            else:
                raise NameError("Vertex not recognized") 
        if type(V_to) == str:
            if "x" in V_to:
                V_to = self.Vp[V_to]
            elif "m" in V_to:
                V_to = self.Vm[V_to]
            else:
                raise NameError("Vertex not recognized")  
        return (V_from, V_to)
    
    def add_edge(self, V_from : Vertex, V_to : Vertex, meas, weight, etype : str=None):
        """Add edge in measurement graph between two nodes.

        Args:
            V_from (Vertex): Source vertex
            V_to (Vertex): Sink Vertex
            meas (numpy array): 3x1 measurement associated with edge
            weight (numpy array): 3x3 weight associated with edge
            etype (str) : type of edge, if specified (p2m_trans, p2p_trans, p2p_rot)
        """
        # Check if inputs are strings and get vertices.    
        V_from, V_to = self.find_edge_verts(V_from, V_to)
        # If source vertex has no edges yet, add to edge dict
        if not V_from in self.E.keys():
            self.E[V_from] = {}
        # Either add edge or update it if already existing
        if not V_to in self.E[V_from].keys():
            self.E[V_from][V_to] = Edge(V_from, V_to, meas, weight, etype)
            # Modify the connection lists in the vertices
            V_from.to_list += [V_to]
            V_to.from_list += [V_from]
        else:
            self.E[V_from][V_to].update(meas, weight, etype)
        
    def get_edge(self, V1, V2):
        """Retrieve an edge

        Args:
            V1 (Vertex, str): outgoing vertex
            V2 ( Vertex, str ): incoming vertex

        Returns:
            Edge : Edge corresponding to vertices if it exists
        """
        # Check if inputs are strings and get vertices.    
        V1, V2 = self.find_edge_verts(V1, V2)
        try:
            e = self.E[V1][V2]
        except:
            e = []
        return e
        