import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

#data
z = 11
x = 13
n_seis =10
v1 = 5          #5km/s
v2 = 5.2        #5.2km/s

#simulate the cross section = 11X13 matrix
velocity_mat = np.zeros((z,x))
velocity_mat[:,:] = v1
#add the anomaly
velocity_mat[1:9,4:7] = v2

#get the slownes matrix sum(1/v(ui) -1/v)
slowness_mat  = 1 / velocity_mat - (1 / v1)

#m vector is the flattened slownes_mat
m_test = np.ravel(slowness_mat)
#plots
fig, axs = plt.subplots(1,2, figsize = (12,6))
sns.heatmap(np.round(velocity_mat,3),annot=True, linewidth=0.5,cmap="Dark2_r", cbar=False,ax=axs[0])
sns.heatmap(np.round(slowness_mat,3)*1e3,annot=True, linewidth=0.5,cmap="YlOrBr", cbar=False,ax=axs[1])
#axs.set_yticklabels(['Earthquake 1','Earthquake 2'])
# axs[0].set_xlabel('Seismograph Position')
# axs[1].set_xlabel('pito')
axs[0].set_xlabel('Km')
axs[0].set_ylabel('Km')
axs[1].set_xlabel('Km')
axs[1].set_ylabel('Km')
axs[0].set_title('Velocity Matrix [Km/s]')
axs[1].set_title('Slownes Matrix [ms/Km]')
plt.show()


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

#create one matrix whit all measurment
t_pure = np.zeros((2,len(t2_pure)))
t_pure[0] = t1_pure
t_pure[1] = t2_pure

#plots
fig, axs = plt.subplots(figsize = (12,6))
axs= sns.heatmap(np.round(t_pure,3),annot=True, linewidth=0.5,cmap="YlGnBu", cbar=False)
axs.set_yticklabels(['Earthquake 1','Earthquake 2'])
axs.set_xlabel('Seismograph Position')
axs.set_title('Time(pure) Anomaly in Seconds')
plt.show()

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







#traces_eq2 = [np.trace(slowness_mat[0:13 - i - 1, 1 + i:13]) for i in range(1, 11)]       # trace of matrix starts from first detector, reduce the matrix 1 row from botom to top, and from left to right until last detector
# #traces_eq2 = [np.trace(slownes_mat[0:i,0:i])for i in range(1,1)]  # flip order of the matrix and get the trace incresing from 2X2 (first detector) to 3X3... until 10X10(last detector)
#
# traces_eq1 = [np.trace(np.fliplr(slowness_mat[0:i, 0:i])) for i in range(2, 12)]  # flip order of the matrix and get the trace incresing from 2X2 (first detector) to 3X3... until 10X10(last detector)
# slownes_eq2 = np.array(traces_eq2)
# slownes_eq1 = np.array(traces_eq1)
# print(np.trace(slowness_mat[0:13 - 0 - 1, 1 + 0:13]))
# print(np.round(slownes_eq1,3))
# print(np.round(slownes_eq2,3))

def G_mat():
  #matrix G is 20 columns, and z*x rows
    G = np.zeros([20,x*z])
    #first 10 rows travel from the left(np.fliplr, flips the columns), first diagoal= np.eye(11, 13, 11-0) from the position of detectors
    for i in range(10):
        G[i] = np.ravel(np.fliplr(np.eye(11, 13, 11-i)))
     #the last 10 are waves traveling from the right and first diagonal =  np.eye(11,13,2+0) from detector position
    for i in range(10,20):
        j = i - 10
        G[i] = np.ravel(np.eye(11,13,2+j))
    return G*math.sqrt(2)
#
# G = math.sqrt(2)*G_mat()
# #simulate the cross section = 11X13 matrix
# velocity_mat = np.zeros((11,13))
# velocity_mat[:11,:] = 5
# velocity_mat[1:9,4:7] = 5.2
#
# #after discrteizing the slownes function, slownes is given by 1/v(u) - 1/v0
# slowness_mat  = 1 / velocity_mat - (1 / 5)
# m = np.ravel(slowness_mat)
# d = G@m
# plt.imshow(G)
# plt.show()
# print(d)
# plt.imshow(slowness_mat)
# plt.show()
#
# #calculate traces, the trace will give the sum of the velocities
#
# traces_eq2 = [np.trace(slowness_mat[0:13 - i - 1, 1 + i:13]) for i in range(1, 11)]       # trace of matrix starts from first detector, reduce the matrix 1 row from botom to top, and from left to right until last detector
# #traces_eq2 = [np.trace(slownes_mat[0:i,0:i])for i in range(1,1)]  # flip order of the matrix and get the trace incresing from 2X2 (first detector) to 3X3... until 10X10(last detector)
#
# traces_eq1 = [np.trace(np.fliplr(slowness_mat[0:i, 0:i])) for i in range(2, 12)]  # flip order of the matrix and get the trace incresing from 2X2 (first detector) to 3X3... until 10X10(last detector)
# slownes_eq2 = np.array(traces_eq2)
# slownes_eq1 = np.array(traces_eq1)
# print(np.trace(slowness_mat[0:13 - 0 - 1, 1 + 0:13]))
# print(np.round(slownes_eq1,3))
# print(np.round(slownes_eq2,3))
# t1_pure = math.sqrt(2)* slownes_eq1
# t2_pure = math.sqrt(2)* slownes_eq2
# noise1 = (1/18)*np.linalg.norm(t1_pure)
# noise2 = (1/18)*np.linalg.norm(t1_pure)
# t1_obs = t1_pure + noise1
# t2_obs = t1_pure + noise2
#
# #plot
# fig, axs = plt.subplots(figsize = (12,6))
# axs= sns.heatmap(np.round(slowness_mat,3),annot=True, linewidth=0.5,cmap="YlGnBu", cbar=False)
# axs.set_xlabel('Km')
# axs.set_ylabel('Km')
# axs.xaxis.set_label_position('top')
# #axs.set_xticks(np.linspace(0,13))
#
#
#
# plt.show()
#formulate the discrete inverse problem







#get them into arrays
# arrival_vel_eq1 = np.array(traces_eq1)
# arrival_vel_eq2 = np.array(traces_eq2)
# #calculate the rays distance
# #the x maximum distance is 12 km and the minimum is 1 km, create an array from 1 to 12
# x_distance = np.linspace(2,11,10)
#
# #rays from earthquake 1 come from the left
# eq1_dist = x_distance*math.sqrt(2)
# # #rays from  earthequake 2 come from the right by symetry just flip(reverse) the order
# eq2_dist=eq1_dist[::-1]
# #
# # #the velocity of the waves without anomaly is 5km/s therefore the arrival time is:
# arrival_time_eq1 = eq1_dist/5
# arrival_time_eq2 = eq2_dist/5
# print(f"""The time of arrival without any anomalies from the first earthquake is {arrival_time_eq1}""")
# print(f"""The time of arrival without any anomalies from the second earthquake is {arrival_time_eq2}""")
# #arrival times with anomaly
# anomaly_arrival_time_eq1 = eq1_dist/arrival_vel_eq1
# anomaly_arrival_time_eq2 = eq2_dist/arrival_vel_eq2
# print(f"""The time of arrival without any anomalies from the first earthquake is {anomaly_arrival_time_eq1}""")
# print(f"""The time of arrival without any anomalies from the second earthquake is {anomaly_arrival_time_eq2}""")
# #adding noise
# noise1 = (1/18)*np.linalg.norm(anomaly_arrival_time_eq1)
# noise2 = (1/18)*np.linalg.norm(anomaly_arrival_time_eq2)
# # #calculate t observed
# t1_obs = anomaly_arrival_time_eq1 + noise1
# t2_obs = anomaly_arrival_time_eq2 + noise2
# print(f"""The observed time of arrival of earth-quake 1 is {t1_obs}""")
# print(f"""The observed time of arrival of earth-quake 2 is {t2_obs}""")
# #formulate the discreate inverese problem
# print(np.trace(velocity_mat[0:12,1:12]))
# print(np.trace(velocity_mat[0:11,2:12]))
# print(traces_eq2)
# print(traces_eq1)


