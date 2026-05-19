
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

def get_T_static():

    # Mesh data
    el_sizef, el_type, dofs_pn = 0.02, 2, 1
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

        # bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_CONVECTION, T_inf)

        
        return (coord, edof, dofs, bdofs, bc, bc_value, element_markers)
    


    
    generate_mesh(show_geometry=False)

    coord, edof, dofs, bdofs, bc, bc_value, element_markers = generate_mesh(show_geometry=False)

    ex,ey = cfc.coord_extract(edof,coord,dofs)
    ndof = np.size(dofs)
    nelem = len(edof)
    ep = [thickness]

    #Stiffness matrix 
    K =  np.zeros((ndof,ndof))

    #Conductivity matrix
    D = np.eye(2)*k

    # Total Load vector
    F_l = np.zeros((ndof, 1))

    # Flow out vector
    F_out = np.zeros((ndof, 1))

    # Convection contribution to load vector
    F_convection = np.zeros((ndof, 1))

    edges = nodesToEdges(bdofs, edof)

    edges_out = edges[MARKER_q_out]

    edges_convection = edges[MARKER_CONVECTION]

    # Calculate flow contribution to load vector
    for edge in edges_out:
        length = np.linalg.norm(coord[edge[1]-1] - coord[edge[0]-1]) #length of edge
        F_out[edge-1] -= q_out * length * thickness / 2 #Form function for flow out, negative since it's leaving the system, divided by 2 since it is a triangle and we have linear shape functions for the edges


    for edge in edges_convection:
        length = np.linalg.norm(coord[edge[1]-1] - coord[edge[0]-1]) #length of edge
        F_convection[edge-1] += alpha_c * thickness * length * T_inf / 2 #

        # Beräkna konvektionsstyvhetsmatrisen för kanten
        Kc_e = (alpha_c * thickness * length / 6.0) * np.array([[2.0, 1.0],[1.0, 2.0]])

        # Put into the global stiffness matrix
        cfc.assem(edge, K, Kc_e)

    for i in range(nelem):
        Ke = cfc.flw2te(ex[i,:],ey[i,:],ep,D)

        # Calculate area, using vector product of vectors and dividing by 2
        x = ex[i, :]
        y = ey[i, :]

        A = 0.5 * np.abs( (x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]))

        F_l_e = (Q*A*thickness/3)* np.array([[1],
                                            [1],
                                            [1]])
        
        cfc.assem(edof[i], K, Ke, F_l, F_l_e)

    F_l += F_out + F_convection #Add flow contribution and convection to load vector
    
    temps, flows = cfc.solveq(K, F_l, bc, bc_value)

    return temps



def get_T_dynamic():

    

    # Mesh data
    el_sizef, el_type, dofs_pn = 0.02, 2, 1
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

    t_tot = 10*60 # Total simulation time in seconds
    n_steps = 1000 # Number of time steps in the simulation

    c_p = 800 #Specific heat
    k = 2 #Thermal conductivity

    thickness = 1 # meter 



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

        # bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_CONVECTION, T_inf)


        
        return (coord, edof, dofs, bdofs, bc, bc_value, element_markers)

    
    coord, edof, dofs, bdofs, bc, bc_value, element_markers = generate_mesh(show_geometry=False)

    ex,ey = cfc.coord_extract(edof,coord,dofs)
    ndof = np.size(dofs)
    nelem = len(edof)
    ep = [thickness]

    #Conductivity matrix
    D = np.eye(2)*k

    edges = nodesToEdges(bdofs, edof)

    edges_out = edges[MARKER_q_out]

    edges_convection = edges[MARKER_CONVECTION]

    # Convection contribution to load vector
    F_convection = np.zeros((ndof, 1))

    #Create C matrix (does not change with time)
    C = np.zeros((ndof, ndof))

    #Stiffness matrix 
    K =  np.zeros((ndof,ndof))

    for i in range(nelem):
        #Assemble C matrix
        Ce = plantml.plantml(ex[i,:], ey[i,:], rho*c_p*thickness)
        cfc.assem(edof[i], C, Ce)

        #Assemble K matrix
        Ke = cfc.flw2te(ex[i,:],ey[i,:],ep,D)
        cfc.assem(edof[i], K, Ke)
        
    for edge in edges_convection:
        length = np.linalg.norm(coord[edge[1]-1] - coord[edge[0]-1]) #length of edge
        F_convection[edge-1] += alpha_c * thickness * length * T_inf / 2 #

        # Beräkna konvektionsstyvhetsmatrisen för kanten
        Kc_e = (alpha_c * thickness * length / 6.0) * np.array([[2.0, 1.0],[1.0, 2.0]])

        # Put into the global stiffness matrix
        cfc.assem(edge, K, Kc_e)

    # Create time array

    t = np.linspace(0, t_tot, n_steps) # Simulate for 10 minutes with 1000 time steps

    delta_t = t[1] - t[0]

    # Initial temperature distribution
    T_n = np.ones((ndof, 1)) * T_inf

    # Create the history of all times
    T_history = np.zeros((ndof, n_steps))
    T_history[:, 0] = T_n.flatten()

    #Iterate over time steps. Recalculating the load vector

    for step in range(1, n_steps):
        time = t[step]

        # Total Load vector
        F_l = np.zeros((ndof, 1))

        # Flow out vector
        F_out = np.zeros((ndof, 1))

        # Note the load vector should be in time step n+1! 

        # Calculate flow contribution to load vector
        for edge in edges_out:
            length = np.linalg.norm(coord[edge[1]-1] - coord[edge[0]-1]) #length of edge
            F_out[edge-1] -= q_out * np.sin(np.pi*(time)/t_tot)* length * thickness / 2 #Form function for flow out, negative since it's leaving the system, divided by 2 since it is a triangle and we have linear shape functions for the edges
        
        for i in range(nelem):
            
            # Calculate area, using vector product of vectors and dividing by 2
            x = ex[i, :]
            y = ey[i, :]

            A = 0.5 * np.abs( (x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]))

            F_l_e = (Q*np.sin(np.pi*(time)/t_tot)*A*thickness/3)* np.array([[1],
                                                [1],
                                                [1]])
            

            # To be able to use assem for load vector
            Ke_fake = np.zeros((3, 3))
            cfc.assem(edof[i], K, Ke_fake, F_l, F_l_e)

        F_l += F_out + F_convection #Add flow contribution and convection to load vector

        T_n = np.linalg.solve(C + delta_t*K, np.matmul(C, T_n) + delta_t*F_l) # Solve for new temperatures
        T_history[:, step] = T_n.flatten()



    return T_history, t

