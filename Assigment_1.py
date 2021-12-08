import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

#%%
#General data
np.random.seed(27)
z = 11
x = 13
n_seis = 10
v1 = 5          #5km/s
v2 = 5.2        #5.2km/s
save_fig = False
verbose = False
cmap_list = ["YlOrBr","Blues", "dark:salmon_r","#69d","seagreen","cubehelix","crest","flare","mako","rocket","Dark2_r","YlGnBu"]
#%%

#%% Functions
#simulate the cross section = 11X13 matrix

def model_anomaly(x1, x2, z1, z2):
    """

    :param x1: int, anomaly coordinate x1
    :param x2: int, anomaly coordinate x2
    :param z1: int, anomaly coordinate z1
    :param z2: int, anomaly coordinate z2
    :return: original model velocity matrix and slowness matrix
    """
    velocity_mat = np.zeros((z, x))
    velocity_mat[:, :] = v1
    # add the anomaly
    velocity_mat[z1:z2, x1:x2] = v2
    # get the slowness matrix sum(1/v(ui) -1/v)
    slowness_mat = 1 / velocity_mat - (1 / v1)
    return slowness_mat, velocity_mat


#calculate t_pure
def calculate_t(slowness_mat,flat_mat=False):
    """

    :param slowness_mat: 11X13 slowness matrix
    :param flat_mat: bool if True returns 1X20 1d t_pure
    :return: 2X10 t_pure matrix
    """
    # earthquake 1 (from the left)
    #flip order of the matrix and get the trace increasing from 2X2 (first detector) to 3X3... until 10X10(last detector)
    tr_eqk1 = [np.trace(np.fliplr(slowness_mat[0:i, 0:i])) for i in range(2, 12)]
    #to obtain the arrival time tr_eqk are multiplied by the distance sqtr(2)
    t1_pure = np.array(tr_eqk1)*math.sqrt(2)
    #earthquake 2 (from the right)
    #calculate the traces with an ofset from colum 2(first detector) to column 10 last detector
    tr_eqk2 =  [np.trace(slowness_mat,offset=i) for i in range(2,12)]
    t2_pure = np.array(tr_eqk2)*math.sqrt(2)
    #create t_pure, one matrix whit all measurment
    t_pure = np.zeros((2,len(t2_pure)))
    t_pure[0] = t1_pure
    t_pure[1] = t2_pure
    if verbose:
        print(f"""The time anomaly in seconds of the first earthquake detected by each seismograph are: \n{t1_pure}""")
        print(f"""The time anomaly in seconds of the second earthquake detected by each seismograph are: \n{t2_pure}""")

    if flat_mat:
        #flattening matrix
        t_pure_1d = np.ravel(t_pure)
        return t_pure_1d
    else:
        return t_pure

#calculate G matrix

#Gmatrix is  a 20X143 matrix, 20 is from the 20 seismographs, and 143 is from 13*11, i.e we have 20 11X20 matrices one for each wave
def G_mat():
    """

    :return: 20X143 G matrix
    """
  #matrix G is 20 columns, and z*x rows
    G = np.zeros([n_seis*2,x*z])
    #first 10 rows travel from the left(np.fliplr, flips the columns), first diagoal= np.eye(11, 13, 11-0) from the position of detectors
    for i in range(n_seis):
        G[i] = np.ravel(np.fliplr(np.eye(z, x, z-i)))
     #the last 10 are waves traveling from the right and first diagonal =  np.eye(11,13,2+0) from detector position
    for i in range(n_seis,n_seis*2):
        j = i - n_seis
        G[i] = np.ravel(np.eye(z,x,2+j))
        #G * distance in each square =sqrt(2)
    return G * math.sqrt(2)

#create noise
def calculate_noise(slowness_mat):
    """
    :param slowness_mat: 11X13 slowness matrix
    :return: noise vector
    """
    #normal distributed with mean value 0
    n = np.random.normal(0,size=2*n_seis)
    #and condition ||n||=||t_pure||/18
    t_pure_1d = calculate_t(slowness_mat,flat_mat=True)
    t_pure_norm = np.linalg.norm(t_pure_1d)
    n_norm = t_pure_norm/18
    #find scaling factor
    y = n_norm/np.linalg.norm(n)
    #create nosise
    n = n * y
    #test
    if verbose:
        print(f"""The noise vector fulfil condition? {round(np.linalg.norm(n),5)==round(n_norm,5)}""")
    return n

def calculate_t_obs(n,slowness_mat,flat_mat=True):
    """

    :param n: noise vector
    :param slowness_mat: 11X13 slowness matrix
    :param flat_mat: returns t_obs as a 1d vector
    :return: t_obs as a 2X10 vector(number of earthquakes X number of detectors)
    """
    t_pure1d = calculate_t(slowness_mat,flat_mat=True)
    t_obs1d = t_pure1d + n
    t_obs = np.zeros((2,n_seis))
    t_obs[0] = t_obs1d[:n_seis]
    t_obs[1]= t_obs1d[n_seis:n_seis*2]
    if flat_mat:
        return t_obs1d
    return t_obs


# find the solution using tikhonov regularization
# build a tikhonov regularization function
def tikhonov_reg(G,d,eps):
    """

    :param G: 20X143 G matrix calculated with G_mat()
    :param d: data parameters d=Gm in this case 1d t_obs
    :param eps: float optimization parameter
    :return: calculated model
    """
    m = np.linalg.inv(G.T@G + np.identity(143)*eps**2)@G.T@d
    return m

def solve_eps(eps,G,d,error):
    """

    :param eps: float optimization parameter calculated with calculate_epsilon()
    :param G: 20X143 G matrix calculated with G_mat()
    :param d: data parameters d=Gm in this case 1d t_obs
    :param error: noise std
    :return: optimization value where epsilon minimize the error
    """
    m = tikhonov_reg(G,d,eps)
    s = np.abs(np.linalg.norm(d - G@m) - d.shape[0]*error**2)

    return s

#calculate epsilon
def calculate_epsilon(n,G,t_obs1d):
    """

    :param n: noise vector
    :param t_obs1d: t_obs
    :return:  list of scanned epsilons and solution, value of epsilon that minimise the function and index
    """
    #data
    error = n.std()
    resolution = 500
    epsilons = np.linspace(1e-2,0.1,resolution)
    solutions = np.zeros(resolution)
    for i, eps in enumerate(epsilons):
        solutions[i] = solve_eps(eps,G,t_obs1d,error)
    #find the epsilon that minimize the soultions
    min_index = np.argmin(solutions)
    min_eps = epsilons[min_index]
    return epsilons, solutions, min_eps, min_index



#%%

#%%plots functions

def models_plots(M1, M2, r1, r2, color1, color2, name1, name2):
    """

    :param M1: Matrix  1
    :param M2: Matrix 2
    :param r1: int, rounding number
    :param r2: int, rounding number
    :param color1: heatmap colors from cmap_list
    :param color2: heatmap colors from cmap_list
    :param name1: 1st figure title
    :param name2: 2nd figure title
    :return: plot of 2 model matrix
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(np.round(M1, r1), annot=True, linewidth=0.5, cmap=color1, cbar=False, ax=axs[0])
    sns.heatmap(np.round(M2, r2), annot=True, linewidth=0.5, cmap=color2, cbar=False, ax=axs[1])
    axs[0].set_xlabel('Km')
    axs[0].set_ylabel('Km')
    axs[1].set_xlabel('Km')
    axs[1].set_ylabel('Km')
    axs[0].set_title(f'{name1}')
    axs[1].set_title(f'{name2}')
    if save_fig:
        plt.savefig(f'model')
    plt.show()


def seismograph_times_plot(M, r,name,color):
    """

    :param M: Seismograph time detection matrix
    :param r: int, rounding value
    :param name: name in the title
    :return: heatmap of time recorded by seismographs
    """
    fig, axs = plt.subplots(figsize = (12,4))
    axs= sns.heatmap(np.round(M,r),annot=True, linewidth=0.5,cmap=color, cbar=False)
    axs.set_yticklabels(['Earthquake 1','Earthquake 2'])
    axs.set_xlabel('Seismograph Position')
    axs.set_title(f'Time({name}) Anomaly in Seconds')
    if save_fig:
        plt.savefig(f'Time({name})')
    plt.show()

#plot epsilon minimization fit
def plot_epsilon(eps,s,min_eps,min_index, name):
    """

    :param eps: range of epsilons
    :param s: solve epsilon solutions
    :param min_eps: minimum epsilon
    :param min_index: minimum epsilon index
    :param name: name to save graph
    :return: Error vs epsilon plot
    """
    fig, ax = plt.subplots(figsize = (10,4))
    ax.plot(eps,s)
    ax.set_xlabel('$\\epsilon$')
    ax.set_ylabel('Error')
    ax.set_title('Epsilon Minimum Fit')
    ax.annotate(f"Min $\\epsilon$ = {min_eps:1.4f}",(eps[min_index+15],s[min_index]))
    if save_fig:
        plt.savefig(f'{name}')
    plt.show()
#%%

#%% run code
def model_simulation(x1,x2,z1,z2):
    #get matrices
    #slowness_mat, velocity_mat = model_anomaly(4,7,1,9)
    slowness_mat, velocity_mat = model_anomaly(x1, x2, z1, z2)
    #plot velocity matrix with slowness
    models_plots(velocity_mat, slowness_mat*1e3, 3, 3, cmap_list[10], cmap_list[0], 'Velocity Matrix [Km/s]', 'Slownes Matrix [ms/Km]')

    #m vector is the flattened slownes_mat
    m_test = np.ravel(slowness_mat)
    #test
    G = G_mat()
    d = G@m_test
    t_pure = calculate_t(slowness_mat)
    test1 = d==np.ravel(t_pure)
    if test1.all():
        print("G matrix is correct")

    #plot sesimograph pure times
    seismograph_times_plot(t_pure, 3,'pure',cmap_list[11])
    # compute noise
    n = calculate_noise(slowness_mat)
    #compute t_obs
    t_obs1d = calculate_t_obs(n,slowness_mat)
    t_obs = calculate_t_obs(n,slowness_mat,flat_mat=False)
    #plot sesimograph observed  times
    seismograph_times_plot(t_obs, 5,'observed',cmap_list[11])
    return slowness_mat,n,G,t_obs1d

# calculate problem with original anomaly x1=4,x2=7,z1=1,z2=9 to get slowness_mat, G matrix, noise and t_obs
slowness_mat, n,G, t_obs1d = model_simulation(4,7,1,9)

#calculate optimal epsilon
epsilons,solutions, min_eps, min_index = calculate_epsilon(n,G,t_obs1d)

#calculate model with optimal epsilon
def model_calculation(G, t_obs1d,min_eps,min_index,slowness_mat,eps_plot=False):
    m = tikhonov_reg(G,t_obs1d,min_eps)
    #plot
    if eps_plot:
        plot_epsilon(epsilons,solutions,min_eps,min_index,'epsilon_vs_error')
    # create a 11X13 m matrix for visualization
    m_mat = m.reshape(z,x)*1e3
    #plot model obtained after tikhonov_reg with real model
    models_plots(m_mat, slowness_mat*1e3, 3, 3, cmap_list[8], cmap_list[0], 'Model Matrix [Km/ms]', 'Slowness Matrix [ms/Km]')
    #
    #calculate t
    t = G@m
    t_mat = t.reshape(2,10)
    # plot seismograph model calculated times
    seismograph_times_plot(t_mat, 5,'calculated',cmap_list[11])
    #test
    t_obs = calculate_t_obs(n,slowness_mat,flat_mat=False)
    test2 = np.round(t,4)==np.round(np.ravel(t_obs),4)
    if test2.all():
        print("m model match input")
    else:
        print('Estimated model does not match input (as was to be expected).')


model_calculation(G, t_obs1d,min_eps,min_index,slowness_mat,eps_plot=True)

#%%test anomaly

# calculate problem with 1X1 anomaly x1=6,x2=7,z1=1,z2=3 to get slowness_mat, G matrix, noise and t_obs
slowness_anomaly, n_anomaly,G_anomaly, t_obs1d_anomaly = model_simulation(6,7,1,2)
model_calculation(G_anomaly, t_obs1d_anomaly,min_eps,min_index,slowness_anomaly)

# calculate problem with 1X6 anomaly x1=6,x2=7,z1=1,z2=3 to get slowness_mat, G matrix, noise and t_obs
slowness_anomaly2, n_anomaly2,G_anomaly2, t_obs1d_anomaly2 = model_simulation(6,7,1,7)
model_calculation(G_anomaly2, t_obs1d_anomaly2,min_eps,min_index,slowness_anomaly2)