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

import proj_temp_results_as_functions as temp_results # Temperatures from a and b part.

# Mesh data
el_sizef, el_type, dofs_pn = 0.02, 2, 2 # 2 dofs per node for structural analysis (x and y displacements)!!!
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

# Calculate von Mises stress from stress components
def vonMisesStress(sigma):
    sigma_xx, sigma_yy, sigma_zz, sigma_xy = sigma[0], sigma[1], sigma[2], sigma[3]
    von_mises = np.sqrt(((sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2 + 6*sigma_xy**2) / 2)
    return von_mises

# B function for plane stress elements, returns the B matrix and the area of the element
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

#Given
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

#Given
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

    # Locking the left edge in x and y direction
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
    enod = (edof[:, 0::2] + 1) // 2 # Get node numbers from edof (integer division by 2 because of 2 dofs per node)
    bnodes = {}
    for key, dof_list in bdofs.items(): # Same for boundary nodes
        bnodes[key] = np.unique((np.array(dof_list) + 1) // 2)

    # List of edges for each boundary condition

    edges = nodesToEdges(bnodes, enod) #total edges on boundary

    edges_p2 = edges[MARKER_p2]

    edges_convection = edges[MARKER_CONVECTION]

    edges_p1 = edges[MARKER_p1]

    #Stiffness matrix 
    K =  np.zeros((ndof,ndof))
    ep = [ptype, thickness]

    F_therm = np.zeros((ndof,1))

    # Extract temperatures from previous tasks 
    T_tot, times = temp_results.get_T_dynamic()
    T_stat = T_tot[:,700]
    #T_stat = temp_results.get_T_static() 

    for i in range(nelem):
        #Assemble K matrix
        Ke = cfc.plante(ex[i,:],ey[i,:],ep,D)
        cfc.assem(edof[i], K, Ke)

        #Thermal force vector
        nod_index = enod[i] - 1
        
        T_element = T_stat[nod_index] 
        T_avg = np.mean(T_element)
        delta_T = T_avg - T_inf  
        
        #Plane strain meaning the z expansion must happen in the x and y direction instead, using Poisson's number
        eps_0 = (1 + v) * alpha * delta_T * np.array([1, 1, 0])
        
        # Make sure it is a numpy array
        D_clean = np.array(D)
        
        # We don't want the z direction
        idx = [0, 1, 3]
        D_2D = D_clean[np.ix_(idx, idx)]
        
        # Stress from thermal expansion, sigma_0 = D * eps_0
        sigma_0 = D_2D @ eps_0


        B, A = Bfunction(ex[i,:], ey[i,:])

        # Integral
        f_therm_e = A * thickness * (B.T @ sigma_0)
        
        # Put in the thermal load vector
        indx = edof[i] - 1
        F_therm[indx, 0] += f_therm_e


    # Total f vector
    F = np.zeros((ndof,1))

    #Boundary  vector
    F_b = np.zeros((ndof,1))

    # Pressure from bottom edge, contribution to load vector
    for edge in edges_p2:
        length = np.linalg.norm(coord[edge[1]-1] - coord[edge[0]-1]) #length of edge

        nod1 = edge[0]
        nod2 = edge[1]

        y1_index = 2*nod1 - 1
        y2_index = 2*nod2 - 1

        F_b[[y1_index, y2_index]] += p2 * thickness * length / 2 # Pressure times area (length*thickness) divided by 2 because of linear shape functions

    # Formula for pressure on the right side, linear variation from p0 at the bottom to 0 at the top
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
    
    plot_mid_x = []
    plot_mid_y = []
    plot_norm_x = []
    plot_norm_y = []

    for edge in edges_convection:
        length = np.linalg.norm(coord[edge[1]-1] - coord[edge[0]-1]) #length of edge
        #vector = coord[edge[1]-1] - coord[edge[0]-1]
        #Rotation 90 degrees clockwise and normalization to get the normal vector pointing outwards
        #normal_vector = np.array([vector[1], -vector[0]]) / np.linalg.norm(vector) 
        mid_point = (coord[edge[1]-1] + coord[edge[0]-1])/2

        mid_y = mid_point[1]
        
        # There is problem with inconsistent orientation of nodes, so we need to check orientation manually.
        # See which hole is closest
        if mid_y < 0.013:   
            center = np.array([0.004, 0.008]) # Hål 1
        elif mid_y < 0.023: 
            center = np.array([0.006, 0.018]) # Hål 2
        elif mid_y < 0.033: 
            center = np.array([0.004, 0.028]) # Hål 3
        else:               
            center = np.array([0.006, 0.038]) # Hål 4

        # Normal vector is from midpoint to center
        normal_vector = (center - mid_point)/np.linalg.norm(center - mid_point)
        
        plot_mid_x.append(mid_point[0])
        plot_mid_y.append(mid_point[1])
        plot_norm_x.append(normal_vector[0])
        plot_norm_y.append(normal_vector[1])

        force_vec = -thickness*length*pc * normal_vector / 2 #Negative since it directed into the battery, divided by 2 because of linear shape functions
        F_1 = force_vec[0]
        F_2 = force_vec[1]

        xindex = 2*edge - 2
        yindex = 2*edge - 1
        F_b[xindex] += F_1
        F_b[yindex] += F_2

    F = F_b + F_therm #Total load vector is the sum of the boundary loads and the thermal loads

    # Solve system
    displacements, reactions = cfc.solveq(K, F, bc, bc_value)

    mises_elements = np.zeros(nelem) # initiate array to store von Mises stress for each element

    # Calculate von Mises stress for each element using the displacements and the thermal expansion, and store in mises_elements
    for i in range(nelem):
        # Extract temperature difference
        nod_index = enod[i] - 1
        T_element = T_stat[nod_index] 
        T_avg = np.mean(T_element)
        delta_T = T_avg - T_inf  

        # Displacements for the element
        displacements_element = displacements[edof[i]-1]
        B,A = Bfunction(ex[i,:], ey[i,:])

        # Calculate the total strain from the displacements, epsilon = B * displacements
        epsilon_almost_fucking_tot = np.matmul(B, displacements_element).flatten()
        # Must be a 4 component vector to be able to subtract the thermal expansion, we just add a 0 for the z component since we are in plane stress
        epsilon_fucking_tot = np.array([epsilon_almost_fucking_tot[0], epsilon_almost_fucking_tot[1], 0, epsilon_almost_fucking_tot[2]])

        # Substract thermal expansion to get the elastic strain
        epsilon_elastic = epsilon_fucking_tot - alpha * delta_T * np.array([1, 1, 1, 0])

        #Hookes law
        sigma_e = D @ epsilon_elastic

        mises_elements[i] = vonMisesStress(sigma_e.flatten())

    # Get nodal von Mises stress by averaging the element von Mises stresses for the elements that share each node
    nodal_mises = elmToNode(mises_elements, enod)

    # Plor in MPa instead of Pa
    nodal_mises_MPa = nodal_mises / 1e6

    # ==========================================
    # VISUELL KONTROLL AV NORMALVEKTORER
    # ==========================================
    plt.figure(figsize=(6, 12))
    
    # Rita upp nätet i bakgrunden så vi ser var pilarna är
    cfv.draw_mesh(
        coords=coord, edof=edof, dofs_per_node=dofs_pn, el_type=el_type, 
        filled=False, color='lightgray'
    )
    
    # RITA PILARNA! (quiver)
    # X, Y (startpunkter) och U, V (riktning). 
    # Ändra 'scale' om pilarna blir för långa/korta (högre siffra = kortare pilar)
    plt.quiver(
        plot_mid_x, plot_mid_y, plot_norm_x, plot_norm_y, 
        color='red', scale=20, width=0.005, label='Normalvektorer'
    )
    
    plt.title("Visuell kontroll av yt-normaler i kylkanaler")
    plt.legend()
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(8, 10))
    
    cfv.draw_nodal_values_shaded(
        nodal_mises_MPa, 
        coords=coord, 
        edof=enod, 
        dofs_per_node=1, 
        el_type=el_type, 
        title="Von Mises stress [MPa] for dynamic temperature distribution at 420s"
    )

    cfv.colorbar(label="Nodal von Mises stress (MPa)") 
    
    plt.axis('equal')
    #plt.savefig("dynamic_von_mises_stress_distribution.png")
    plt.show()

    # Plot displacements, magnified to be visible
    
    # Scaling factor for displacements to make them visible in the plot
    mag_factor = 100.0 

    plt.figure(figsize=(8, 10))
    
    cfv.draw_mesh(
        coords=coord, edof=edof, dofs_per_node=dofs_pn, el_type=el_type, 
        filled=False, color='lightgray'
    )
    
    cfv.draw_displacements(
        a=displacements, coords=coord, edof=edof, dofs_per_node=dofs_pn, el_type=el_type,
        draw_undisplaced_mesh=False, title=f"Displacements static temperature (scaling factor: {mag_factor}x)",
        color='red', magnfac=mag_factor
    )
    
    plt.axis('equal')

    #plt.savefig("static_displacement_magnified.png")
    plt.show()


    
   