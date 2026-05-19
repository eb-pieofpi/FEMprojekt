import calfem.geometry as cfg
import calfem.mesh as cfm

import calfem.utils as cfu

import calfem.core as cfc

import calfem.vis_mpl as cfv

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import plantml

mpl.use('TkAgg')

import calfem.vis_mpl as cfv

import numpy as np

import proj_temp_results_as_functions as temp_results

# Mesh data
el_sizef, el_type, dofs_pn = 0.01, 2, 2
mesh_dir = "."

# boundary markers
MARKER_SYMMETRY=0
MARKER_CONVECTION=1
MARKER_p2 = 2
MARKER_p1 = 3
MARKER_NADA = 4

#Variables

alpha_c = 50 #Convection coeff
T_inf = 293 #Surrounding temp

q_out = 2000 # flow out W/m^2
Q = 4e5 # Heat generation W/m^3
E = 30e9 #Youngs modulus Pa
v = 0.3 #Poisson
G = E/(2*(1+v)) #Shear modulus
alpha = 8e-6 #thermal expansion coeff
rho = 1500 #Density
sigma_limit = 150e6 # Yield strength Pa

t_tot = 10*60 # Total simulation time in seconds
n_steps = 1000 # Number of time steps in the simulation

c_p = 800 #Specific heat
k = 2 #Thermal conductivity

thickness = 1 # meter 

p0 = 1e8 # pressure for p1
p2 = 1e7 # Pressure
pc = 1e6 # Contact pressure

def vonMisesStress(sigma):
    sigma_xx, sigma_yy, sigma_zz, sigma_xy = sigma[0], sigma[1], sigma[2], sigma[3]
    von_mises = np.sqrt(((sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2 + 6*sigma_xy**2) / 2)
    return von_mises

def Bfunction(ex, ey):
    x1, x2, x3 = ex[0], ex[1], ex[2]
    y1, y2, y3 = ey[0], ey[1], ey[2]

    # Arean (A)
    A = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    # Geometriska konstanter
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    # B-matrisen (Strain-Displacement matrix)
    B = (1.0 / (2.0 * A)) * np.array([
        [b1,  0, b2,  0, b3,  0],
        [ 0, c1,  0, c2,  0, c3],
        [c1, b1, c2, b2, c3, b3]
    ])

    return B, A

def nodesToEdges ( nodes : dict , enod : np . array ) -> dict :
    """ Returns a list of edges given nodes
        Args :
    nodes ( np . array ) : nodes on boundary
    enod ( np . array ) : element connectivity matrix
        Returns :
    list : edges on boundary
     """
    # Initialize edges dict
    edges = {}
    for key in nodes . keys () :
        edges [ key ] = []

    for con in zip ( enod ) :
        for key in nodes . keys () :
            I = np . intersect1d ( con , nodes [ key ])
            if len ( I ) == 2:
                edges [ key ]. append ( I )
            
    return edges

def elmToNode ( eV : np . array , edof : np . array ) -> np . array :
    """ Estimates nodal values from element - based values
    Args :
    eV ( np . array ) : element values
    edof ( np . array ) : element connectivity matrix
    Returns :
    np . array : nodal - based values
     """

    nnod : int = np . max ( edof )
    ne : int = 0
    nV = np . zeros (( nnod ,) )
    # Loop over nodes
    for n in range (0 , nnod ) :
        ne = 0
        # Check which elements contain the node
        for e , eldof in enumerate ( edof ) :
        # If e contains the node add the elemental value
            if (( n +1) in eldof ) :
                ne += 1
                nV [ n ] += eV [ e ]
            # Divide by total number of elements
        nV [ n ] /= ne

    return nV


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
    g.spline([1, 2], 1, marker=MARKER_p2) 
    g.spline([2, 3], 2, marker=MARKER_p1)
    g.spline([3, 4], 3, marker=MARKER_NADA)
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

    for dof in bdofs[MARKER_SYMMETRY]:
        
        bc = np.append(bc, dof)
        bc_value = np.append(bc_value, 0.0)

   
    # bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_CONVECTION, T_inf)


    
    return (coord, edof, dofs, bdofs, bc, bc_value, element_markers)

if __name__=="__main__":
    coord, edof, dofs, bdofs, bc, bc_value, element_markers = generate_mesh(show_geometry=False)

    ex,ey = cfc.coord_extract(edof,coord,dofs)
    ndof = np.size(dofs)
    nelem = len(edof)
    
    # Constitutive matrix
    ptype = 2
    D = cfc.hooke(ptype, E, v)

    # We cant use bdofs and edof anymore since we need nodes and we have two dofs per node.
    enod = (edof[:, 0::2] + 1) // 2
    bnodes = {}
    for key, dof_list in bdofs.items():
        bnodes[key] = np.unique((np.array(dof_list) + 1) // 2)

    # List of edges for each boundary condition

    edges = nodesToEdges(bnodes, enod) #total edges on boundary

    edges_p2 = edges[MARKER_p2]

    edges_convection = edges[MARKER_CONVECTION]

    edges_p1 = edges[MARKER_p1]
    ep = [ptype, thickness]



    T_tot, times = temp_results.get_T_dynamic()
    
            #Boundary  vector
    F_b = np.zeros((ndof,1))

    for edge in edges_p2:
        length = np.linalg.norm(coord[edge[1]-1] - coord[edge[0]-1]) #length of edge

        nod1 = edge[0]
        nod2 = edge[1]

        y1_index = 2*nod1 - 1
        y2_index = 2*nod2 - 1

        F_b[[y1_index, y2_index]] += p2 * thickness * length / 2 # Pressure times area (length*thickness) divided by 2 because of linear shape functions


    p1 = lambda y: (y/0.023-1)*p0
    for edge in edges_p1:
        length = np.linalg.norm(coord[edge[1]-1] - coord[edge[0]-1]) #length of edge
        
        # y coordinates to be able to calculate the pressure at the edge
        y1 = coord[edge[0]-1][1]
        y2 = coord[edge[1]-1][1]

        p_1 = p1(y1)
        p_2 = p1(y2)

        F_1 = length*thickness*(2*p_1 + p_2)/6 # Pressure times area (length*thickness) times the shape function contribution for a linear edge
        F_2 = length*thickness*(p_1 + 2*p_2)/6

        # Find degree of freedom for the x direction for the two nodes of the edge
        nod1 = edge[0]
        nod2 = edge[1]

        x1_index = 2*nod1 - 2
        x2_index = 2*nod2 - 2

        F_b[x1_index] += F_1
        F_b[x2_index] += F_2
    
    for edge in edges_convection:
        length = np.linalg.norm(coord[edge[1]-1] - coord[edge[0]-1]) #length of edge
        vector = coord[edge[1]-1] - coord[edge[0]-1]

        #Rotation 90 degrees clockwise and normalization to get the normal vector pointing outwards
        normal_vector = np.array([vector[1], -vector[0]]) / np.linalg.norm(vector) 

        force_vec = -thickness*length*pc * normal_vector / 2
        F_1 = force_vec[0]
        F_2 = force_vec[1]

        xindex = 2*edge - 2
        yindex = 2*edge - 1
        F_b[xindex] += F_1
        F_b[yindex] += F_2

    #Stiffness matrix 
    K =  np.zeros((ndof,ndof))
    
    for i in range(nelem):
            #Assemble K matrix
            Ke = cfc.plante(ex[i,:],ey[i,:],ep,D)
            cfc.assem(edof[i], K, Ke)

    time_subset = times[::100] # Ta var 10:e tid för att minska antalet plottade punkter

    max_mises = np.zeros(len(time_subset))

    for j in range(0, len(times), 100):

        F_therm = np.zeros((ndof,1))

        print(f"Calculating for time step {j+1}/{len(T_tot)}")

        T_stat = T_tot[:,j]
        for i in range(nelem):

            #Thermal force vector
            nod_index = enod[i] - 1
            
            T_element = T_stat[nod_index] 
            T_avg = np.mean(T_element)
            delta_T = T_avg - T_inf  
            
            # Eftersom vi bygger B-matrisen i 2D måste vi baka in Poissons tal (1+v)
            # för att simulera att z-riktningen är fastlåst (Plan töjning).
            eps_0 = (1 + v) * alpha * delta_T * np.array([1, 1, 0])
            
            # FIXEN: Tvätta bort CALFEMs gamla "np.matrix"-format!
            D_clean = np.array(D)
            
            # Klipp ut 3x3 från den RENA matrisen
            idx = [0, 1, 3]
            D_2D = D_clean[np.ix_(idx, idx)]
            
            # Nu blir detta en perfekt 1D-vektor med 3 element
            sigma_0 = D_2D @ eps_0

            B, A = Bfunction(ex[i,:], ey[i,:])

            # Nodkrafterna! f = A * t * B^T * sigma_0 (Använder @ för korrekt matris-matte)
            f_therm_e = A * thickness * (B.T @ sigma_0)
            
            # Montera in krafterna i totala lastvektorn!
            indx = edof[i] - 1
            F_therm[indx, 0] += f_therm_e


        # Total f vector
        F = np.zeros((ndof,1))

        F = F_b + F_therm

        displacements, reactions = cfc.solveq(K, F, bc, bc_value)

        mises_elements = np.zeros(nelem)


        for i in range(nelem):
            nod_index = enod[i] - 1
            T_element = T_stat[nod_index] 
            T_avg = np.mean(T_element)
            delta_T = T_avg - T_inf  
            
            displacements_element = displacements[edof[i]-1]
            B,A = Bfunction(ex[i,:], ey[i,:])

            epsilon_almost_fucking_tot = np.matmul(B, displacements_element).flatten()
            epsilon_fucking_tot = np.array([epsilon_almost_fucking_tot[0], epsilon_almost_fucking_tot[1], 0, epsilon_almost_fucking_tot[2]])

            epsilon_elastic = epsilon_fucking_tot - alpha * delta_T * np.array([1, 1, 1, 0])

            sigma_e = D @ epsilon_elastic

            mises_elements[i] = vonMisesStress(sigma_e.flatten())

        nodal_mises = elmToNode(mises_elements, enod)

        nodal_mises_MPa = nodal_mises / 1e6

        max_mises[j//100] = max(nodal_mises_MPa)


        # plt.figure(figsize=(8, 10))
        

        # cfv.draw_nodal_values_shaded(
        #     nodal_mises_MPa, 
        #     coords=coord, 
        #     edof=enod, 
        #     dofs_per_node=1, 
        #     el_type=el_type, 
        #     title="von Mises stress [MPa] for static temperature distribution"

        # )
        
        # cfv.colorbar(label="Nodal von Mises stress (MPa)") 
        

        # plt.axis('equal')
        # #plt.savefig("static_von_mises_stress_distribution.png")
        # plt.show()

    plt.plot(time_subset, max_mises)
    plt.xlabel("Time (s)")
    plt.ylabel("Max von Mises stress (MPa)")
    plt.title("Max von Mises stress vs time")
    plt.grid()
    plt.savefig("max_von_mises_vs_time2.png")
    plt.show()
        

