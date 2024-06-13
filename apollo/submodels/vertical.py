import numpy as np
from numpy.typing import NDArray

"""
// Computes a 71-layer T-P profile from a layer-by-layer input profile
void Planet::getProfLayer(vector<double> tpprofile)
{
  tprof = tpprofile;
  prprof = vector<double>(nlevel);

  for(int i=0; i<nlevel; i++){
    prprof[i] = pow(10,minP + (maxP-minP)*i/(nlevel-1));
  }

  vector<double> dpdr(nlevel);

  hprof.back() = 0.;

  for(int i=nlevel-2; i>=0; i--){
    dpdr[i] = G*mp*mu/k/tprof[i]/(rp+hprof[i+1])/(rp+hprof[i+1]);
    if(hprof[i+1]>=rp*5.0 || isinf(hprof[i+1])){
      hprof[i] = rp*5.0;
    }
    else{
      hprof[i] = hprof[i+1] + log(prprof[i+1]/prprof[i])/dpdr[i];
      if(hprof[i]>=rp*5.0 || isinf(hprof[i])) hprof[i] = rp*5.0;
    }
  }
}
// end getProfLayer
"""


def hydrostatic_equilibrium(
    pressure: NDArray[np.float_],
    temperature: NDArray[np.float_],
    mass: float,
    radius: float,
) -> NDArray[np.float_]:
    pass
