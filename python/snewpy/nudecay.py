from pylab import *
from scipy.integrate import quad, quad_vec
from scipy.special import gammaincc,gamma

def lightest_integrand(x, eps, rbar, emean, alpha, rinit=0):
    if alpha <= 2:
        raise ValueError(f"Ill-defined results for alpha <= 2 (here alpha={alpha})")
    return x**(alpha-2) * (1 - exp(-(rbar - rinit)/(x*eps))) * exp(-(1+alpha)*x*eps/emean)


def lightest_integral(rbar, eps, emean, alpha, rinit=0):
    return quad_vec(lambda x: lightest_integrand(x,eps,rbar,emean,alpha,rinit), 1, 40)[0]

# The lightest eigenstate is fed by the decays of the heaviest state
# 0 = electron, 1 = mu, tau
def lightest_eigenstate(rbar, eps, lumi, emean, alpha, zeta, rinit=0):
    Ni0 = (alpha[0]+1)**(alpha[0]+1)/(emean[0] * gamma(alpha[0]+1))/emean[0]
    Ni1 = (alpha[1]+1)**(alpha[1]+1)/(emean[1] * gamma(alpha[1]+1))/emean[1]
    finit = lumi[1] * Ni1 * (eps/emean[1])**alpha[1] * exp(-(alpha[1]+1)*eps/emean[1])
    # print(Ni0, Ni1, finit, eps, emean[0], alpha[0], rbar, rinit)
    # print(finit[30], (2*zeta*Ni0*lumi[0]* (eps/emean[0])**alpha[0] * lightest_integral(rbar,eps,emean[0],alpha[0],rinit))[30])
    return 2*zeta*Ni0*lumi[0]* (eps/emean[0])**alpha[0] * lightest_integral(rbar,eps,emean[0],alpha[0],rinit) + finit


# The heaviest eigenstate decays away
def heaviest_eigenstate(rbar, eps, lumi, emean, alpha, rinit=0):
    Ni0 = (alpha[0]+1)**(alpha[0]+1)/(emean[0] * gamma(alpha[0]+1))/emean[0]
    f0 = lumi[0] * Ni0 * (eps/emean[0])**alpha[0] * exp(-(alpha[0]+1)*eps/emean[0])
    # print("heavy: ", f0[30], exp(-(rbar - rinit)/eps)[30])
    return f0*exp(-(rbar - rinit)/eps)

# The eigenstate in the middle does not evolve in vacuum
def middle_eigenstate(eps, lumi, emean, alpha):
    Ni1 = (alpha[1]+1)**(alpha[1]+1)/(emean[1] * gamma(alpha[1]+1))/emean[1]
    return lumi[1] * Ni1 * (eps/emean[1])**alpha[1] * exp(-(alpha[1]+1)*eps/emean[1])

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
