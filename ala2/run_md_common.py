import numpy as np
def dihedral_indices(top):
    a = {(x.residue.name, x.name): x.index for x in top.atoms()}
    return ([a[("ACE","C")], a[("ALA","N")], a[("ALA","CA")], a[("ALA","C")]], [a[("ALA","N")], a[("ALA","CA")], a[("ALA","C")], a[("NME","N")]],
            [a[("ACE","CH3")], a[("ACE","C")], a[("ALA","N")], a[("ALA","CA")]], [a[("ALA","CA")], a[("ALA","C")], a[("NME","N")], a[("NME","C")]])
def dihedral(x, idx):
    p0, p1, p2, p3 = x[idx]; b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2; b1n = b1 / np.linalg.norm(b1)
    v, w = b0 - b0.dot(b1n) * b1n, b2 - b2.dot(b1n) * b1n
    return np.arctan2(np.cross(b1n, v).dot(w), v.dot(w))
