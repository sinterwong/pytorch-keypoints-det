import scipy.io as scio

dataFile = 'Data_0000046.mat'
data = scio.loadmat(dataFile)

print(data.keys())
print("data['pos']: ", data['pos'])
print("data['hmap']: ", data['hmap'])
print("data['segmap']: ", data['segmap'])
print("data['handPos']: ", data['handPos'])
print("data['depth']: ", data['depth'])