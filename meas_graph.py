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
    
class MeasGraph:
    """
    Class defining the measurement (factor) graph for a given problem
    """
    # Pose Vertex Set - dictionary indexed on vertex labels
    Vp = {}
    # Map or Landmark Vertex Set - dictionary indexed on vertex labels
    Vm = {}
    # Edge Structure
    # dict of dicts indexed on vertex sets
    # Stores measurement and weights
    E = {}
    
    def __init__(self, r_p_in0, C_p_0, r_m_in0):
        """ Constructor for measurement graph in a SLAM problem or subproblem.
        Expects ground truth input values.

        Args:
            r_p_in0: (list) of translations to pose frame from world frame in world frame
            C_p_0:   (list) of spatial math rotation matrices rotating base frame vector to pose frame
            r_m_in0: (list) of translations to map point in world frame
        """
        # Make sure that pose length data is consistent
        if not len(C_p_0) == len(r_p_in0):
            raise Exception("Input Pose data has inconsistent length")
        # Loop through poses and define vertices
        for i in range(len(r_p_in0)):
            # Check shape
            assert(r_p_in0[i].shape == (3,1), f"Pose translation {i} not (3,1)")
            assert(C_p_0[i].shape == (3,3),f"Pose Rotation {i} not (3,3)")
            # Add to list
            label = f'x{i}'
            self.Vp[f'x{i}'] = Vertex(label, r_p_in0[i], C_p_0[i])
        # Loop through map points and define vertices
        for i in range(len(r_m_in0)):
            # Check shape
            assert(r_m_in0[i].shape == (3,1),f"Map translation {i} not (3,1)")
            label = f'm{i}'
            self.Vm[label] = MapVertex(label,r_m_in0[i])
         
    def addEdge(self, V_from, V_to, meas, weight):
        """Add edge in measurement graph between two nodes.

        Args:
            V_from (Vertex): Source vertex
            V_to (Vertex): Sink Vertex
            meas (numpy array): 3x1 measurement associated with edge
            weight (numpy array): 3x3 weight associated with edge
        """
        # If just strings given, convert to vertices
        if type(V_from) == str:
            if V_from in self.Vp.keys():
                V_from = self.Vp[V_from]
            else:
                raise Exception(f"FROM index {V_from} is not in the list of pose indices")
        if type(V_to) == str:
            if V_to in self.Vp.keys():
                V_to = self.Vp[V_to]
            elif V_to in self.Vm.keys():
                V_to = self.Vm[V_to]
            else:
                raise Exception(f"TO index {V_to} is not in the list of pose or map indices")
        # Modify the connection lists in the vertices
        V_from.to_list += [V_to]
        V_to.from_list += [V_from]
        # If source vertex has no edges yet, add to edge dict
        if not V_from in self.E.keys():
            self.E[V_from] = {}
        # Define edge
        self.E[V_from][V_to] = (meas, weight)