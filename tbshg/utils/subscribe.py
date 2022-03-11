
def number_indice_to_string(index)->str:
    """
    Convert a number to a string.
    """
    indicename = ["x", "y", "z"]
    return "".join([indicename[i] for i in index])