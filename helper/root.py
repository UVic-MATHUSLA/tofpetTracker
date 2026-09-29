import warnings
from array import array
import ROOT
import numpy as np

def GetBinCenters(h):
    Nbins = h.GetNbinsX()
    bincenters = np.array([])
    for i in np.arange(1,Nbins+1):
        i = int(i)
        bincenters = np.append(bincenters,h.GetBinCenter(i))
    return bincenters

def GetBinEdges(h):
    Nbins = h.GetNbinsX()
    bin_edges_low = np.array([])
    bin_widths = np.array([])
    for i in np.arange(1, Nbins+1):
        i = int(i)
        bin_edges_low = np.append(bin_edges_low, h.GetBinLowEdge(i))
        bin_widths = np.append(bin_widths, h.GetBinWidth(i))
    bin_edges_low = np.append(bin_edges_low, bin_edges_low[-1]+bin_widths[-1])
    return bin_edges_low

def GetBinErrors(h):
    Nbins = h.GetNbinsX()
    errs = np.array([])
    for i in np.arange(1, Nbins+1):
        i = int(i)
        errs = np.append(errs, h.GetBinError(i))
    return errs

def GetBinContents(h):
    Nbins = h.GetNbinsX()
    n = np.array([])
    for i in np.arange(1, Nbins+1):
        i = int(i)
        n = np.append(n, h.GetBinContent(i))
    return n

def get_info(h):
    """
    Return the bin content, bin edges and bin errors of a histogram
    """
    n, ibins, errs = GetBinContents(h), GetBinEdges(h), GetBinErrors(h)
    return n, ibins, errs

class tfile_reader:
    """
    Helper calss for ROOT TFile.
    It wraps the mostly used funcitons to open and read objects from ROOT file.

    tfile = ROOT.TFile.Open(fname_digi)
    tree_name = tfile.GetListOfKeys()[1].GetName()
    Tree = tfile.Get(tree_name)
    branches = [Tree.GetListOfBranches()[i].GetName() for i in range(len(Tree.GetListOfBranches()))]
    entries = Tree.GetEntries()
    """
    def __init__(self, filename, verbose = 0, option = ""):
        self.file = ROOT.TFile(filename, option)

    @property
    def keys(self):
        self._keys = self.file.GetListOfKeys()
        return self._keys

    def ls(self):
        print(self.file.ls())

    def get(self, key):
        self.object = self.file.Get(key)
        return self.object

    def vector2list(self, vector):
        return [vector.at(i) for i in range(vector.size())]

    def get_tree(self, key):
        self.tree = self.file.Get(key)
        self.entries = self.tree.GetEntries()
        self.current_entry = -1

        # Read the branch name and type
        # Make a dictionary for each branch
        self.branches = {}
        self.branches_disabled = []
        self.data = {}
        for branch in self.tree.GetListOfBranches():
            # Get the leaf for the branch
            leaf = branch.GetLeaf(branch.GetName())
            # Get the data type of the leaf
            data_type = leaf.GetTypeName()
            
            self.branches[branch.GetName()] = [data_type]
        return self.tree


    def ls_tree(self):
        print("Entries:", self.entries)
        for b in self.branches:
            print(f"Branch: {b:<20}, Type: {self.branches[b]}")

    def disable_branches(self, branches):
        pass
    
    def enable_branches(self, branches):
        pass
            
    def get_entry(self, entry) -> dict:
        self.tree.GetEntry(entry)

        for b in self.branches:
            if "vector" in self.branches[b][0]:
                self.data[b] = self.vector2list(getattr(self.tree, b))
            else:
                self.data[b] = getattr(self.tree, b)
                
        return self.data


    def make_tree(self, branches):
        """
        Input:
        ---
        branches: list of pair [[branch_name1, branch_type1, branch_]]
        """
        return 0
    
class tfile_writer:
    """
    Supported types: (int, float, double, bool, vector<int>, vector<float>, vector<double>, vector<int64>)
    
    Example usage:
    ```
    # Example usage:
    # Create a new ROOT tree writer
    tree_writer = tfile_writer("my_tree", "output.root")
    
    # Define branches
    tree_writer.define_branch("energy", "float")
    tree_writer.define_branch("event_id", 'int')
    tree_writer.define_branch("hit_x", 'vector<float>')
    
    # Fill the tree with some data
    data = {"energy": 42.5,
            "event_id": 1,
            "hit_x": [1,2,3,4,5]}
    tree_writer.fill(data)
    
    # Write the tree to the file and close it
    tree_writer.write_and_close()

    # Read it back to check
    f1 = root.tfile_reader("output.root")
    f1.ls()
    f1.get_tree("my_tree")
    f1.ls_tree()
    f1.get_entry(0)
    ```
    """
    
    def __init__(self, tree_name, file_name):
        self.file = ROOT.TFile(file_name, "RECREATE")
        self.tree = ROOT.TTree(tree_name, "Tree created by RootTreeWriter")
        self.branch_buffers = {} # dict of {branch_name: branch_buffer}
        self.branch_types = {} # dict of {branch_name: branch_type_string}

    def define_branch(self, branch_name, branch_type):
        """
        Define a branch in the tree and set up the corresponding buffer.

        :param branch_name: Name of the branch.
        :param branch_type: Type of the branch (e.g., int, float, double, bool, vector<int>, vector<float>, vector<double>, vector<int64>).
        """
        if branch_type == 'int':
            buffer = array('i', [0])
            self.tree.Branch(branch_name, buffer, branch_name+"/I")
        elif branch_type == 'float':
            buffer = array('f', [0])
            self.tree.Branch(branch_name, buffer, branch_name+"/F")
        elif branch_type == 'double':
            buffer = array('d', [0])
            self.tree.Branch(branch_name, buffer, branch_name+"/D")
        elif branch_type == 'b':
            buffer = array('i', [0])
            self.tree.Branch(branch_name, buffer, branch_name+"/O")
        elif branch_type == 'vector<int>':
            buffer = ROOT.vector('int')()
            self.tree.Branch(branch_name, buffer)
        elif branch_type == 'vector<float>':
            buffer = ROOT.vector('float')()
            self.tree.Branch(branch_name, buffer)
        elif branch_type == 'vector<double>':
            buffer = ROOT.vector('double')()
            self.tree.Branch(branch_name, buffer)
        elif branch_type == 'vector<int64>':
            buffer = ROOT.vector('Long64_t')()
            self.tree.Branch(branch_name, buffer)
        elif branch_type == 'vector<string>':
            buffer = ROOT.std.vector('string')()
            self.tree.Branch(branch_name, buffer)
        else:
            raise ValueError(f"Unsupported branch type: {branch_type}")

        self.branch_buffers[branch_name] = buffer
        self.branch_types[branch_name] = branch_type

    def fill(self, data_dict):
        """
        Fill the tree with data from a Python dictionary.

        :param data_dict: Dictionary where keys are branch names and values are the data to be written.
        The value of each key should match the type specified before. A list is expected for vector types.
        """
        for branch_name, buffer in self.branch_buffers.items():
            if branch_name not in data_dict:
                warnings.warn(f"Data for branch '{branch_name}' not provided in the dictionary.")
                continue
                
            branch_type = self.branch_types[branch_name]

            if "vector" in branch_type:
                buffer.clear()
                for val in data_dict[branch_name]:
                    buffer.push_back(val.item() if isinstance(val, np.generic) else val)
            else:
                value = data_dict[branch_name]
                buffer[0] = value.item() if isinstance(value, np.generic) else value
                
        self.tree.Fill()

    def fill_all(self, data_dict_all):
        """
        Fill all entries with a python dictionary

        :param data_dict: Dictionary where keys are branch names and values are the data to be written.
        Each key is a LIST of single objects that match the type specified before. 
        For vector type, it should be a list of list.
        """
        for i in range(len(data_dict_all[list(data_dict_all.keys())[0]])):
            dict_temp = {key:val[i] for key, val in data_dict_all.items()}
            self.fill(dict_temp)        
        

    def write_and_close(self):
        """
        Write the tree to the file and close the file.
        """
        self.file.cd()
        self.tree.Write()
        self.file.Close()
