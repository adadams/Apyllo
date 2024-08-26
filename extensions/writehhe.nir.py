import numpy as np

fh2 = open('h2only.nir.dat','r')
fhe = open('he.nir.dat','r')
fhhe = open('h2he.nir.dat','w')

h2lines = fh2.readlines()
helines = fhe.readlines()

npress = 18
pmin = -6.0
pmax = 2.5
ntemp = 36
tmin = 1.875061263
tmax = 3.599199194
#tmax = 3.549938111

#wmin = 0.97
#wmax = 1.13
#resolv = 50000.
wmin = 0.6
wmax = 5.0
resolv = 10000.
nspec = (int)(np.ceil(np.log(wmax/wmin)*resolv))+1
print(nspec)

print('{0:d} {1:f} {2:f} {3:d} {4:f} {5:f} {6:d} {7:f} {8:f} {9:f}\n'.format(npress,pmin,pmax,ntemp,tmin,tmax,nspec,wmin,wmax,resolv))
#fhhe.write('{0:d} {1:f} {2:f} {3:d} {4:f} {5:f} {6:d} {7:f} {8:f} {9:f}\n'.format(npress,pmin,pmax,ntemp,tmin,tmax,nspec,wmin,wmax,resolv))

for i in range(1,649):
    h2l = h2lines[i].split()
    hel = helines[i].split()
    for j in range(0,nspec):
        h2 = float(h2l[j])
        he = float(hel[j])
        #if(i%100==0 and j%1000==0): print h2,he
        hhe = h2*0.86 + he*0.14
        fhhe.write('{0:12.5e} '.format(hhe))
    fhhe.write('\n')
