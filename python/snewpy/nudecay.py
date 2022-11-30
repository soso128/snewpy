from pylab import *
from scipy.integrate import quad, quad_vec
from scipy.special import gammaincc,gamma

def lightest_integrand(x, eps, rbar, emean, alpha, rinit=0, flip=False):
    if alpha <= 2:
        raise ValueError(f"Ill-defined results for alpha <= 2 (here alpha={alpha})")
    res = x**(alpha-2) * (1 - exp(-(rbar - rinit)/(x*eps))) * exp(-(1+alpha)*x*eps/emean)
    if flip:
        return res * (x - 1)
    else:
        return res


def lightest_integral(rbar, eps, emean, alpha, rinit=0, flip=False):
    return quad_vec(lambda x: lightest_integrand(x,eps,rbar,emean,alpha,rinit,flip=flip), 1, 40)[0]

def pinched_spectrum(eps, lumi, emean, alpha):
    Ni = (alpha+1)**(alpha+1)/(emean * gamma(alpha+1))/emean
    return lumi * Ni * (eps/emean)**alpha * exp(-(alpha+1)*eps/emean)

def decay_spectrum(rbar, eps, lumi, emean, alpha, rinit=0, flip=False):
    Ni = (alpha+1)**(alpha+1)/(emean * gamma(alpha+1))/emean
    return Ni*lumi* (eps/emean)**alpha * lightest_integral(rbar,eps,emean,alpha,rinit,flip=flip)

# The lightest eigenstate is fed by the decays of the heaviest state
# If Dirac, [0] = heaviest, [1] = lightest
# If Majorana, [0] = helicity conserving, [1] = helicity flipping, [2] = lightest state
def lightest_eigenstate(rbar, eps, lumi, emean, alpha, zeta, rinit=0, flip=False):
    Ni0 = (alpha[0]+1)**(alpha[0]+1)/(emean[0] * gamma(alpha[0]+1))/emean[0]
    finit = pinched_spectrum(eps, lumi[-1], emean[-1], alpha[-1])
    if len(lumi) == 2:
        zeta_coeff = 1 - zeta if flip else zeta
        return 2*zeta_coeff * sum(decay_spectrum(rbar, eps, ll, em, alph, flip=flip) for ll,em,alph in zip(lumi[:-1],emean[:-1],alpha[:-1])) + finit
    else:
        not_flipped = 2*zeta*decay_spectrum(rbar, eps, lumi[0], emean[0], alpha[0])
        flipped = 2*(1-zeta)*decay_spectrum(rbar, eps, lumi[1], emean[1], alpha[1], flip=True)
        return finit + flipped + not_flipped

# The heaviest eigenstate decays away
def heaviest_eigenstate(rbar, eps, lumi, emean, alpha, rinit=0, majorana=False):
    f0 = pinched_spectrum(eps, lumi, emean, alpha)
    return f0*exp(-(2 if majorana else 1) * (rbar - rinit)/eps) 

# The eigenstate in the middle does not evolve in vacuum
def middle_eigenstate(eps, lumi, emean, alpha):
    return pinched_spectrum(eps, lumi, emean, alpha)

def final_states(rbar, eps, lumi, emean, alpha, zeta, normal_ordering=True, rinit=0):
    heavy = heaviest_eigenstate(rbar, eps, lumi, emean, alpha, rinit)
    light = lightest_eigenstate(rbar, eps, lumi, emean, alpha, zeta, rinit)
    medium = middle_eigenstate(eps, lumi, emean, alpha)
    theta12 = 33.44*pi/180
    theta13 = 8.57 * pi/180
    Ue3_2 = sin(theta13)**2
    Ue2_2 = sin(theta12)**2 * cos(theta13)**2
    Ue1_1 = cos(theta12)**2 * cos(theta13)**2
    if normal_ordering:
        fe_final = Ue3_2 * heavy + Ue2_2 * medium + Ue1_2 * light
        fx_final = (1-Ue3_2) * heavy + (1-Ue2_2) * medium + (1-Ue1_2) * light
        return [fe_final, fx_final]
    else:
        fe_final = Ue2_2 * heavy + Ue1_2 * medium + Ue3_2 * light
        fx_final = (1-Ue2_2) * heavy + (1-Ue1_2) * medium + (1-Ue3_2) * light
        return [fe_final, fx_final]
