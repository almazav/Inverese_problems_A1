import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

#data
z = 11
x = 13
n_seis = 10
v1 = 5          #5km/s
v2 = 5.2        #5.2km/s
cmap_list = ["YlOrBr","Blues", "dark:salmon_r","#69d","seagreen","cubehelix","crest","flare","mako","rocket","Dark2_r","YlGnBu"]

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
    # get the slownes matrix sum(1/v(ui) -1/v)
    slowness_mat = 1 / velocity_mat - (1 / v1)
    return slowness_mat, velocity_mat

#get matrices
slowness_mat, velocity_mat = model_anomaly(4,7,1,9)

#m vector is the flattened slownes_mat
m_test = np.ravel(slowness_mat)

#calculate t_pure

# earthquake 1 (from the left)
#flip order of the matrix and get the trace increasing from 2X2 (first detector) to 3X3... until 10X10(last detector)
tr_eqk1 = [np.trace(np.fliplr(slowness_mat[0:i, 0:i])) for i in range(2, 12)]
#to obtain the arrival time tr_eqk are multiplied by the distance sqtr(2)
t1_pure = np.array(tr_eqk1)*math.sqrt(2)
print(f"""The time anomaly in seconds of the first earthquake detected by each seismograph are: \n{t1_pure}""")
#earthquake 2 (from the right)
#calculate the traces with an ofset from colum 2(first detector) to column 10 last detector
tr_eqk2 =  [np.trace(slowness_mat,offset=i) for i in range(2,12)]
t2_pure = np.array(tr_eqk2)*math.sqrt(2)
print(f"""The time anomaly in seconds of the second earthquake detected by each seismograph are: \n{t2_pure}""")
#create t_pure, one matrix whit all measurment
t_pure = np.zeros((2,len(t2_pure)))
t_pure[0] = t1_pure
t_pure[1] = t2_pure
t_pure_1d = np.ravel(t_pure)

#calculate G matrix

#Gmatrix is  a 20X143 matrix, 20 is from the 20 seismographs, and 143 is from 13*11, i.e we have 20 11X20 matrices one for each wave
def G_mat():

  #matrix G is 20 columns, and z*x rows
    G = np.zeros([n_seis*2,x*z])
    #first 10 rows travel from the left(np.fliplr, flips the columns), first diagoal= np.eye(11, 13, 11-0) from the position of detectors
    for i in range(n_seis):
        G[i] = np.ravel(np.fliplr(np.eye(z, x, z-i)))
     #the last 10 are waves traveling from the right and first diagonal =  np.eye(11,13,2+0) from detector position
    for i in range(n_seis,n_seis*2):
        j = i - n_seis
        G[i] = np.ravel(np.eye(z,x,2+j))
    return G * math.sqrt(2)
#test
G = G_mat()
d = G@m_test
test1 = d==np.ravel(t_pure)
if test1.all():
    print("G matrix is correct")


#create noise

#normal distributed with mean value 0
n = np.random.normal(0,size=2*n_seis)
#and condition ||n||=||t_pure||/18
t_pure_norm = np.linalg.norm(t_pure_1d)
n_norm = t_pure_norm/18
#find scaling factor
y = n_norm/np.linalg.norm(n)
#create nosise
n = n * y
#test
print(f"""The noise vector fulfil condition? {round(np.linalg.norm(n),5)==round(n_norm,5)}""")

#add noise to t_pure to create t_obs

t_obs_1d = t_pure_1d + n
t_obs = np.zeros((2,n_seis))
t_obs[0] = t_obs_1d[:n_seis]
t_obs[1]= t_obs_1d[n_seis:n_seis*2]


# find the solution using tikhonov regularization

# build a tikhonov regularization function
def tikhonov_reg(G,d,eps):
    m = np.linalg.inv(G.T@G + np.identity(143)*eps**2)@G.T@d
    return m

def solve_eps(eps,G,d,error):
    m = tikhonov_reg(G,d,eps)
    s = np.abs(np.linalg.norm(d - G@m) - d.shape[0]*error**2)
    return s
#data
error = n.std()
resolution = 500
epsilons = np.linspace(1e-2,0.1,resolution)
solutions = np.zeros(resolution)
for i, eps in enumerate(epsilons):
    solutions[i] = solve_eps(eps,G,t_obs_1d,error)

min_index = np.argmin(solutions)
min_eps = epsilons[min_index]


#get model matrix
m = tikhonov_reg(G,t_obs_1d,min_eps)
# create a 11X13 m matrix for visualization
m_mat = m.reshape(z,x)*1e3

#calculate t
t = G@m
t_mat = t.reshape(2,10)
#test
test2 = np.round(t,4)==np.round(np.ravel(t_obs),4)
if test2.all():
    print("m model is correct")
else:
    print('FAIL!!!!')

#plots

def models_plots(M1, M2, r1, r2, color1, color2, name1, name2, save_fig=False):
    """

    :param M1: Matrix  1
    :param M2: Matrix 2
    :param r1: int, rounding number
    :param r2: int, rounding number
    :param color1: heatmap colors from cmap_list
    :param color2: heatmap colors from cmap_list
    :param name1: 1st figure title
    :param name2: 1st figure title
    :param save_fig: if true save figure
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
        plt.savefig(f'{name1}_{name2}')
    plt.show()


def seismograph_times_plot(M, r,name,color, save_fig=False):
    """

    :param M: Seismograph time detection matrix
    :param r: int, rounding value
    :param name: name in the title
    :return: heatmap of time recorded by seismographs
    """
    fig, axs = plt.subplots(figsize = (12,6))
    axs= sns.heatmap(np.round(M,r),annot=True, linewidth=0.5,cmap=color, cbar=False)
    axs.set_yticklabels(['Earthquake 1','Earthquake 2'])
    axs.set_xlabel('Seismograph Position')
    axs.set_title(f'Time({name}) Anomaly in Seconds')
    if save_fig:
        plt.savefig(f'Time({name}')
    plt.show()

#plot velocity matrix with slowness
models_plots(velocity_mat, slowness_mat*1e3, 3, 3, cmap_list[10], cmap_list[0], 'Velocity Matrix [Km/s]', 'Slownes Matrix [ms/Km]', save_fig=False)

#plot sesimograph pure times
seismograph_times_plot(t_pure, 3,'pure',cmap_list[11])

#plot sesimograph observed  times
seismograph_times_plot(t_obs, 5,'observed',cmap_list[11])


#plot epsilon minimization fit
fig, ax = plt.subplots(figsize = (12,6))
ax.plot(epsilons,solutions)
ax.set_xlabel('$\\epsilon$')
ax.set_ylabel('Error')
ax.set_title('Epsilon Minimum Fit')
ax.annotate(f"Min $\\epsilon$ = {min_eps:1.4f}",(min_eps,solutions[min_index]))
plt.show()

#plot model obtained after tikhonov_reg with real model
models_plots(m_mat, slowness_mat*1e3, 3, 3, cmap_list[8], cmap_list[0], 'Model Matrix [Km/ms]', 'Slownes Matrix [ms/Km]', save_fig=False)

#plot sesimograph model calculated  times
seismograph_times_plot(t_mat, 5,'calculated',cmap_list[11])
