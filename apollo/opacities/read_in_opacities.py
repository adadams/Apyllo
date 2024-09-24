"""
// Reads in the opacities tables
void Planet::readopac(vector<int> mollist, vector<double> wavens, string table, string opacdir){
  string specfile;

  string gaslist[19] = {"h2he","h2","he","h-","h2o","ch4","co","co2","nh3","h2s","Burrows_alk","Lupu_alk","crh","feh","tio","vo","hcn","n2","ph3"};

  if(table=="hires"){

    // Gray atmosphere for testing purposes.
    if(nspec==0){
      int x;
      for(int j=0; j<npress; j++){
        for(int k=0; k<ntemp; k++){
          for(int l=0; l<nwave; l++){
            x = (int)(l/degrade);
            mastertable[j][k][x][0] = 6.629e-24;
          }
        }
      }
    } // end if(nspec==0)

    else{
      for(int i=0; i<nspec; i++){
        int index = mollist[i];
        int x;
        double val;

        if(gaslist[index]=="h-"){
          printf("%s computed\n",gaslist[index].c_str());
          for(int j=0; j<npress; j++){
            for(int k=0; k<ntemp; k++){
              for(int l=0; l<nwave; l++){
                double wn = wmin*exp(l/res);
                x = (int)(l/degrade);
                double tmid = tmin*pow(10,k/deltalogt);
                double bf = HminBoundFree(tmid, wn);
                double ff = HminFreeFree(tmid, wn);
                val = bf + ff;
                if(isnan(val) || isinf(val) || val < 1.e-50) val = 1.e-50;
                if(x<ntable) mastertable[j][k][x][i] += val/(double)degrade;
              }
            }
          }
        } // end if(gaslist[index]=="h-")

        else{
          specfile = opacdir + "/gases/" + gaslist[index] + "." + hires + ".dat";
          printf("%s\n",specfile.c_str());

          ifstream opacin(specfile.c_str());
          if(!opacin) cout << "Opacity File Not Found" << std::endl;
          double temp;
          opacin >> temp >> temp >> temp >> temp >> temp >> temp >> temp >> temp >> temp >> temp;

          for(int j=0; j<npress; j++){
            for(int k=0; k<ntemp; k++){
              for(int l=0; l<nwave; l++){
                x = (int)(l/degrade);
                opacin >> val;
                if(isnan(val) || isinf(val) || val < 1.e-50) val = 1.e-50;
                if(x<ntable) mastertable[j][k][x][i] += val/(double)degrade;
              }
            }
          }
          opacin.close();
        } // end else(gaslist)
      } // end for
    } // end else(nspec)
  } // end if(table=="hires")

  if(table=="lores"){

    // Gray atmosphere for testing purposes.
    if(nspec==0){
      for(int j=0; j<npress; j++){
        for(int k=0; k<ntemp; k++){
          for(int l=0; l<nwavelo; l++){
            lotable[j][k][l][0] = 6.629e-24;
          }
        }
      }
    } // end if(nspec==0)

    else{
      for(int i=0; i<nspec; i++){
        int index = mollist[i];
        double val;

        if(gaslist[index]=="h-"){
          printf("%s computed\n",gaslist[index].c_str());
          for(int j=0; j<npress; j++){
            for(int k=0; k<ntemp; k++){
              for(int l=0; l<nwavelo; l++){
                double wn = wmin*exp(l/reslo);
                double tmid = tmin*pow(10,k/deltalogt);
                double bf = HminBoundFree(tmid, wn);
                double ff = HminFreeFree(tmid, wn);
                val = bf + ff;
                if(isnan(val) || isinf(val) || val < 1.e-50) val = 1.e-50;
                lotable[j][k][l][i] = val;
              }
            }
          }
        } // end if(gaslist[index]=="h-")

        else{
          specfile = opacdir + "/gases/" + gaslist[index] + "." + lores + ".dat";
          printf("%s\n",specfile.c_str());

          ifstream opacin(specfile.c_str());
          if(!opacin) cout << "Opacity File Not Found" << std::endl;
          double temp;
          opacin >> temp >> temp >> temp >> temp >> temp >> temp >> temp >> temp >> temp >> temp;

          for(int j=0; j<npress; j++){
            for(int k=0; k<ntemp; k++){
              for(int l=0; l<nwavelo; l++){
                opacin >> val;
                if(isnan(val) || isinf(val) || val < 1.e-50) val = 1.e-50;
                lotable[j][k][l][i] = val;
              }
            }
          }
          opacin.close();
        } // end else(gaslist)
      } // end for
    } // end else(nspec)
  } // end if(table=="lores")

  return;
}
// end readopac
"""
