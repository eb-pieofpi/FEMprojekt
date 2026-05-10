import calfem.geometry as cfg
import calfem.mesh as cfm

import calfem.utils as cfu

import calfem.core as cfc

import calfem.vis_mpl as cfv

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

import calfem.vis_mpl as cfv

import numpy as np

# Mesh data
el_sizef, el_type, dofs_pn = 0.05, 2, 1
mesh_dir = "."

# boundary markers
MARKER_SYMMETRY=0
MARKER_CONVECTION=1
MARKER_q_out = 2

#Variables

alpha_c = 50 #Convection coeff
T_inf = 293 #Surrounding temp

q_out = 2000 # flow out W/m^2
Q = 4e5 # Heat generation W/m^3
E = 30e9 #Youngs modulus Pa
v = 0.3 #Poisson
alpha = 8e-6 #thermal expansion coeff
rho = 1500 #Density

c_p = 800 #Specific heat
k = 2 #Thermal conductivity

thickness = 1 # meter 

def generate_mesh(show_geometry: bool):
    # initialize mesh
    g = cfg.geometry()
    
    # define parameters
    R = 0.002
    H = 0.046
    W = 0.01

    # add points
    g.point([0, 0], 1) #Low left corner
    g.point([W, 0], 2) #Low right corner
    g.point([W, H], 3) #High right corner
    g.point([0, H], 4) #High left corner

    #Circle 1 points
    g.point([0.004, 0.008], 40) # Center
    g.point([0.004+R, 0.008], 41) # Right of circle
    g.point([0.004, 0.008+R], 42) # Top of circle
    g.point([0.004-R, 0.008], 43) # Left of circle
    g.point([0.004, 0.008-R], 44) # Bottom of circle

    #Circle 2 points
    g.point([0.006, 0.018], 50) # Center 
    g.point([0.006+R, 0.018], 51) # Right of circle
    g.point([0.006, 0.018+R], 52) # Top of circle
    g.point([0.006-R, 0.018], 53) # Left of circle
    g.point([0.006, 0.018-R], 54) # Bottom of circle

    #Circle 3 points
    g.point([0.004, 0.028], 60) # Center 
    g.point([0.004+R, 0.028], 61) # Right of circle
    g.point([0.004, 0.028+R], 62) # Top of circle
    g.point([0.004-R, 0.028], 63) # Left of circle
    g.point([0.004, 0.028-R], 64) # Bottom of circle

    #Circle 4 points
    g.point([0.006, 0.038], 70) # Center 
    g.point([0.006+R, 0.038], 71) # Right of circle
    g.point([0.006, 0.038+R], 72) # Top of circle
    g.point([0.006-R, 0.038], 73) # Left of circle
    g.point([0.006, 0.038-R], 74) # Bottom of circle
    
    #Create circles

    g.circle([41, 40, 42], 41, marker=MARKER_CONVECTION) #1 top right quarter 
    g.circle([42, 40, 43], 42, marker=MARKER_CONVECTION) #1 top left quarter 
    g.circle([43, 40, 44], 43, marker=MARKER_CONVECTION) #1 bottom left quarter 
    g.circle([44, 40, 41], 44, marker=MARKER_CONVECTION) #1 bottom right quarter

    g.circle([51, 50, 52], 51, marker=MARKER_CONVECTION) #2 top right quarter 
    g.circle([52, 50, 53], 52, marker=MARKER_CONVECTION) #2 top left quarter 
    g.circle([53, 50, 54], 53, marker=MARKER_CONVECTION) #2 bottom left quarter 
    g.circle([54, 50, 51], 54, marker=MARKER_CONVECTION) #2 bottom right quarter

    g.circle([61, 60, 62], 61, marker=MARKER_CONVECTION) #3 top right quarter 
    g.circle([62, 60, 63], 62, marker=MARKER_CONVECTION) #3 top left quarter 
    g.circle([63, 60, 64], 63, marker=MARKER_CONVECTION) #3 bottom left quarter 
    g.circle([64, 60, 61], 64, marker=MARKER_CONVECTION) #3 bottom right quarter

    g.circle([71, 70, 72], 71, marker=MARKER_CONVECTION) #1 top right quarter 
    g.circle([72, 70, 73], 72, marker=MARKER_CONVECTION) #1 top left quarter 
    g.circle([73, 70, 74], 73, marker=MARKER_CONVECTION) #1 bottom left quarter 
    g.circle([74, 70, 71], 74, marker=MARKER_CONVECTION) #1 bottom right quarter

    # Lines 
    g.spline([1, 2], 1, marker=MARKER_q_out) 
    g.spline([2, 3], 2, marker=MARKER_SYMMETRY)
    g.spline([3, 4], 3, marker=MARKER_CONVECTION)
    g.spline([4, 1], 4, marker=MARKER_SYMMETRY)

    # define surface
    g.surface([1, 2, 3, 4], holes = [[41,42,43,44],[51,52,53,54],[61,62,63,64],[71,72,73,74]])

    # generate mesh
    mesh = cfm.GmshMeshGenerator(g, mesh_dir=mesh_dir)
    mesh.el_size_factor = el_sizef
    mesh.el_type = el_type
    mesh.dofs_per_node = dofs_pn
    coord, edof, dofs, bdofs, element_markers = mesh.create()

    # display mesh
    if show_geometry:
        fig, ax = plt.subplots()

        cfv.draw_geometry(
            g,
            label_curves=True,
            title="Geometry: Projectx"
        )
        
        cfv.draw_mesh(
            coords=coord, 
            edof=edof, 
            dofs_per_node=dofs_pn, 
            el_type=el_type, 
            filled=False, # Sätt till True om du vill ha ifyllda trianglar
            title="Mesh: Projectx"
        )
        
        plt.show()


    # Boundary Conditions
    bc, bc_value = np.array([], 'i'), np.array([], 'f')

    bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_CONVECTION, T_inf)


    
    return (coord, edof, dofs, bdofs, bc, bc_value, element_markers)

if __name__=="__main__":
    coord, edof, dofs, bdofs, bc, bc_value, element_markers = generate_mesh(show_geometry=True)

    ex,ey = cfc.coord_extract(edof,coord,dofs)
    ndof = np.size(dofs)
    nelem = len(edof)
    ep = [thickness]

    #Stiffness matrix 
    K =  np.zeros((ndof,ndof))

    #Conductivity matrix
    D = np.eye(2)*k

    for i in range(nelem):
        Ke = cfc.flw2te(ex[i,:],ey[i,:],ep,D)
        cfc.assem(edof[i], K, Ke)

    # Load vector
    F_l = np.zeros((ndof, 1))


    for i in range(nelem):
        
        # Calculate area, using vector product of vectors and dividing by 2
        x = ex[i, :]
        y = ey[i, :]

        A = 0.5 * np.abs( (x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]))

        F_l_e = (Q*A*thickness/3)* np.array([[1],
                                             [1],
                                             [1]])
        F_l = cfc.assem(edof[i], F_l, F_l_e)




