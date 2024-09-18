#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main()
{
  int dlen = 8;
  int recnum = 0;
  int address = dlen*recnum;
  double headers[9];
  double val;
  string buffer;
  char* s = new char[100];
  ifstream ifile("h2spec.dat");
  ofstream ofile;
  ofile.open("h2only.50k.dat");
  double k = 1.38064852e-16;
  double amagat = 2.6867805e19;  // weird units for h2,he

  //for(int n=0; n<9; n++) ifile >> headers[n];
  vector<vector<double> > h2grid(138,vector<double>(1000,0));
  for(int j=0; j<138; j++){
    for(int k=0; k<1000; k++){
      ifile >> s;
      val = atof(s);
      if(j==20 && k==10000) printf("%e\n",val);
      if(isnan(val)) val = -33.0;
      val = pow(10,val);
      h2grid[j][k] = val;
      //ofile.write(reinterpret_cast<char*>(&val),sizeof(double));
      //if(k==0) printf("%d %d %d %e\n",i,j,k,val);
    }
  }

  // Y-band: 0.97-1.13
  // J-band: 1.17-1.33
  // H-band: 1.49-1.78
  // K-band: 1.99-2.31
  
  double lmin = 0.6;
  double lmax = 2.6;
  double resolv = 50000.;
  int nwave = (int)(ceil(log(lmax/lmin)*resolv))+1;
  
  vector<vector<vector<double> > > h2prof(18,vector<vector<double> >(36,vector<double>(nwave,0)));
  // Important values for h2+he
  double wmin = 0.;
  double wmax = 19980.;
  int numwave = 1000;
  double tmin = 75.;     // linear in temperature
  double tmax = 4000.;   // pressure is irrelevant
  int numtemp = 138;

  double res = (wmax-wmin)/(numwave-1.);

  double* logtemp = new double[numtemp];
  for(int i=0; i<numtemp; i++) logtemp[i] = 75. + 25.*i;

  vector<double> wavens(nwave,0);
  for(int i=0; i<nwave; i++) wavens[i] = 10000./(lmin*exp(i/resolv));
  vector<double> tprof(36,0);
  for(int i=0; i<36; i++){
    tprof[i] = 75. * pow(10,i/20.3) * 0.999999;
    printf("%d %f\n",i,tprof[i]);
  }
  // interpolation variables
  double opr1, opr2, opac1, opac2, opac;
  double h2he;
  double* xsec = new double[4];
  string fname;
    
  int* ad = new int[4];
  
  for(int m=0; m<nwave; m++){
    double deltaw = (wavens[m]-wmin)/(wmax-wmin)*(numwave-1.);
    int jw = (int)deltaw;
    if(jw<0) jw=0;
    if(jw>numwave-2) jw=numwave-2;
    double w1i = wmin + jw*res;
    double w2i = wmin + (jw+1.)*res;
    double dwi = (wavens[m]-w1i)/(w1i-w2i);
    
    for(int i=0; i<36; i++){

      // compute interpolation indices
      double tl = tprof[i];
      double deltat = (tl-tmin)/(tmax-tmin)*(numtemp-1.);
      int jt = (int)deltat;
      if(jt>numtemp-2) jt = numtemp-2;
      double dti = (tl-logtemp[jt])/(logtemp[jt+1]-logtemp[jt]);
      if(jt<0){
	jt=0.;
	dti=0.;
      }
      if(jt>numtemp-1){
	jt=numtemp-2;
	dti=0.999999;
      }

      xsec[0] = h2grid[jt][jw];
      xsec[1] = h2grid[jt][jw+1];
      xsec[2] = h2grid[jt+1][jw];
      xsec[3] = h2grid[jt+1][jw+1];
      
      // interpolate
      opr1 = xsec[0] + dti*(xsec[2]-xsec[0]);
      opr2 = xsec[1] + dti*(xsec[3]-xsec[1]);
      opac = opr1 + dwi*(opr2-opr1);
      //if(m%5000==0 && i%15==0) printf("%d %d %e %e %e\n",m,i,xsec[0],opr1,opac);
      if(m==10000 && i==29) printf("%e\n",opac);
      
      // opac is given in cm^-1 amg^-2
      opac /= amagat*amagat;
      //if(m==10000 && i==29) printf("%e %e\n",opac,amagat);
      // convert to cm^5 molecule^-2
      
      // unit conversion
      for(int j=0; j<18; j++){
	double pressure = pow(10,0.5*j); // pressure in cgs
	double nmol = pressure/k/tprof[i]; // compute number density
	double tempopac = opac * nmol;   // convert to cm^2 molecule^-1

	if(m==10000 && i==29) printf("%e %e %e %e\n",opac,pressure,nmol,tempopac);
	// This is the cross section, and a higher density results in a larger cross section.
	
	//double namg = (pressure/k/tprof[i])/(1013250./k/273.15); // compute number density in amg
	//double tempopac = opac*namg*namg; //convert to opacity per particle in units of cm^-1 amg^-2
	//tempopac /= pow(1013250./k/273.15,2); // convert to cm^2 per particle
	//if(m%5000==0 && i%15==0) printf("%d %d %d %e %e %e\n",m,i,j,pressure,namg,tempopac);
	//if(waven<10000) printf("%e %e\n",namg,opac);
	h2prof[j][i][m] = tempopac;
      }
    }
  }

  
  for(int i=0; i<18; i++){
    for(int j=0; j<36; j++){
      for(int m=0; m<nwave; m++){
	ofile << h2prof[i][j][m] << " ";
	//if(m%5000==0 && j%15==0) printf("%d %d %d %e\n",i,j,m,h2prof[i][j][m]);
      }
      ofile << std::endl;
    }
  }
  
}
