import os
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
from astropy import units as u
from astropy.table import Table, join
from astropy.units.quantity import Quantity
from scipy.special import loggamma

from snewpy.neutrino import Flavor
from functools import wraps

from snewpy.neutrino import MassHierarchy, MixingParameters
import snewpy.nudecay as nd


def _wrap_init(init, check):
    @wraps(init)
    def _wrapper(self, *arg, **kwargs):
        init(self, *arg, **kwargs)
        check(self)
    return _wrapper


class SupernovaModel(ABC):
    """Base class defining an interface to a supernova model."""

    def __init_subclass__(cls, **kwargs):
        """Hook to modify the subclasses on creation"""
        super().__init_subclass__(**kwargs)
        cls.__init__ = _wrap_init(cls.__init__, cls.__post_init_check)

    def __init__(self, time, metadata):
        """Initialize supernova model base class
        (call this method in the subclass constructor as ``super().__init__(time,metadata)``).

        Parameters
        ----------
        time : ndarray of astropy.Quantity
            Time points where the model flux is defined.
            Must be array of :class:`Quantity`, with units convertable to "second".
        metadata : dict
            Dict of model parameters <name>:<value>,
            to be used for printing table in :meth:`__repr__` and :meth:`_repr_markdown_`
        """
        self.time = time
        self.metadata = metadata
        
    def __repr__(self):
        """Default representation of the model.
        """

        mod = f"{self.__class__.__name__} Model"
        try:
            mod += f': {self.filename}'
        except AttributeError:
            pass
        s = [mod]
        for name, v in self.metadata.items():
            s += [f"{name:16} : {v}"]
        return '\n'.join(s)

    def __post_init_check(self):
        """A function to check model integrity after initialization"""
        clsname = self.__class__.__name__
        try:
            t = self.time
            m = self.metadata
        except AttributeError as e:
            clsname = self.__class__.__name__
            raise TypeError(f"Model not initialized. Please call 'SupernovaModel.__init__' within the '{clsname}.__init__'") from e

    def _repr_markdown_(self):
        """Markdown representation of the model, for Jupyter notebooks.
        """
        mod = f'**{self.__class__.__name__} Model**'
        try:
            mod +=f': {self.filename}'
        except:
            pass
        s = [mod,'']
        if self.metadata:
            s += ['|Parameter|Value|',
                  '|:--------|:----:|']
            for name, v in self.metadata.items():
                try: 
                    s += [f"|{name} | ${v.value:g}$ {v.unit:latex}|"]
                except:
                    s += [f"|{name} | {v} |"]
        return '\n'.join(s)

    def get_time(self):
        """Returns
        -------
        ndarray of astropy.Quantity
            Snapshot times from the simulation
        """
        return self.time

    @abstractmethod
    def get_initial_spectra(self, t, E, flavors=Flavor):
        """Get neutrino spectra at the source.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial spectra.
        flavors: iterable of snewpy.neutrino.Flavor
            Return spectra for these flavors only (default: all)

        Returns
        -------
        initialspectra : dict
            Dictionary of neutrino spectra, keyed by neutrino flavor.
        """
        pass

    def get_initialspectra(self, *args):
        """DO NOT USE! Only for backward compatibility!

        :meta private:
        """
        warn("Please use `get_initial_spectra()` instead of `get_initialspectra()`!", FutureWarning)
        return self.get_initial_spectra(*args)

    def get_transformed_spectra(self, t, E, flavor_xform):
        """Get neutrino spectra after applying oscillation.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial and oscillated spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial and oscillated spectra.
        flavor_xform : FlavorTransformation
            An instance from the flavor_transformation module.

        Returns
        -------
        dict
            Dictionary of transformed spectra, keyed by neutrino flavor.
        """
        initialspectra = self.get_initial_spectra(t, E)
        transformed_spectra = {}

        transformed_spectra[Flavor.NU_E] = \
            flavor_xform.prob_ee(t, E) * initialspectra[Flavor.NU_E] + \
            flavor_xform.prob_ex(t, E) * initialspectra[Flavor.NU_X]

        transformed_spectra[Flavor.NU_X] = \
            flavor_xform.prob_xe(t, E) * initialspectra[Flavor.NU_E] + \
            flavor_xform.prob_xx(t, E) * initialspectra[Flavor.NU_X] 

        transformed_spectra[Flavor.NU_E_BAR] = \
            flavor_xform.prob_eebar(t, E) * initialspectra[Flavor.NU_E_BAR] + \
            flavor_xform.prob_exbar(t, E) * initialspectra[Flavor.NU_X_BAR]

        transformed_spectra[Flavor.NU_X_BAR] = \
            flavor_xform.prob_xebar(t, E) * initialspectra[Flavor.NU_E_BAR] + \
            flavor_xform.prob_xxbar(t, E) * initialspectra[Flavor.NU_X_BAR] 

        return transformed_spectra   


    def get_oscillatedspectra(self, *args):
        """DO NOT USE! Only for backward compatibility!

        :meta private:
        """
        warn("Please use `get_transformed_spectra()` instead of `get_oscillatedspectra()`!", FutureWarning)
        return self.get_transformed_spectra(*args)

def get_value(x):
    """If quantity x has is an astropy Quantity with units, return just the
    value.

    Parameters
    ----------
    x : Quantity, float, or ndarray
        Input quantity.

    Returns
    -------
    value : float or ndarray
    
    :meta private:
    """
    if type(x) == Quantity:
        return x.value
    return x

def get_decayed_eigenstates(rbar, zeta, En, lumi, emean, alpha, ecoeff, model="phi0"):
    heavy, medium, light, aheavy, amedium, alight = None, None, None, None, None, None
    if model == "phi0":
        heavy = nd.heaviest_eigenstate(rbar, En, lumi[0], emean[0], alpha[0])/ecoeff**2 / (u.erg * u.s)
        medium = nd.middle_eigenstate(En, lumi[2], emean[2], alpha[2])/ecoeff**2 / (u.erg * u.s)
        light = nd.lightest_eigenstate(rbar, En, lumi[[0,2]], emean[[0,2]], alpha[[0,2]], zeta)/ecoeff**2 / (u.erg * u.s)
        aheavy = nd.heaviest_eigenstate(rbar, En, lumi[3], emean[3], alpha[3])/ecoeff**2 / (u.erg * u.s)
        amedium = nd.middle_eigenstate(En, lumi[3], emean[3], alpha[3])/ecoeff**2 / (u.erg * u.s)
        alight = nd.lightest_eigenstate(rbar, En, lumi[[3,1]], emean[[3,1]], alpha[[3,1]], zeta)/ecoeff**2 / (u.erg * u.s)
    if model == "phi2":
        heavy = nd.heaviest_eigenstate(rbar, En, lumi[0], emean[0], alpha[0])/ecoeff**2 / (u.erg * u.s)
        medium = nd.middle_eigenstate(En, lumi[2], emean[2], alpha[2])/ecoeff**2 / (u.erg * u.s)
        light = nd.lightest_eigenstate(rbar, En, lumi[[3,2]], emean[[3,2]], alpha[[3,2]], zeta, flip=True)/ecoeff**2 / (u.erg * u.s)
        aheavy = nd.heaviest_eigenstate(rbar, En, lumi[3], emean[3], alpha[3])/ecoeff**2 / (u.erg * u.s)
        amedium = nd.middle_eigenstate(En, lumi[3], emean[3], alpha[3])/ecoeff**2 / (u.erg * u.s)
        alight = nd.lightest_eigenstate(rbar, En, lumi[[0,1]], emean[[0,1]], alpha[[0,1]], zeta, flip=True)/ecoeff**2 / (u.erg * u.s)
    if model == "majorana":
        heavy = nd.heaviest_eigenstate(rbar, En, lumi[0], emean[0], alpha[0], majorana=True)/ecoeff**2 / (u.erg * u.s)
        medium = nd.middle_eigenstate(En, lumi[2], emean[2], alpha[2])/ecoeff**2 / (u.erg * u.s)
        light = nd.lightest_eigenstate(rbar, En, lumi[[0,3,2]], emean[[0,3,2]], alpha[[0,3,2]], zeta)/ecoeff**2 / (u.erg * u.s)
        aheavy = nd.heaviest_eigenstate(rbar, En, lumi[3], emean[3], alpha[3], majorana=True)/ecoeff**2 / (u.erg * u.s)
        amedium = nd.middle_eigenstate(En, lumi[3], emean[3], alpha[3])/ecoeff**2 / (u.erg * u.s)
        alight = nd.lightest_eigenstate(rbar, En, lumi[[3,0,3]], emean[[3,0,3]], alpha[[3,0,3]], zeta)/ecoeff**2 / (u.erg * u.s)
    return heavy, medium, light, aheavy, amedium, alight

class PinchedModel(SupernovaModel):
    """Subclass that contains spectra/luminosity pinches"""
    def __init__(self, simtab, metadata):
        """ Initialize the PinchedModel using the data from the given table.

        Parameters
        ----------
        simtab: astropy.Table 
            Should contain columns TIME, {L,E,ALPHA}_NU_{E,E_BAR,X,X_BAR}
            The values for X_BAR may be missing, then NU_X data will be used
        metadata: dict
            Model parameters dict
        """
        if not 'L_NU_X_BAR' in simtab.colnames:
            # table only contains NU_E, NU_E_BAR, and NU_X, so double up
            # the use of NU_X for NU_X_BAR.
            for val in ['L','E','ALPHA']:
                simtab[f'{val}_NU_X_BAR'] = simtab[f'{val}_NU_X']
        # Get grid of model times.
        time = simtab['TIME'] << u.s
        # Set up dictionary of luminosity, mean energy and shape parameter
        # alpha, keyed by neutrino flavor (NU_E, NU_X, NU_E_BAR, NU_X_BAR).
        self.luminosity = {}
        self.meanE = {}
        self.pinch = {}
        for f in Flavor:
            self.luminosity[f] = simtab[f'L_{f.name}'] << u.erg/u.s
            self.meanE[f] = simtab[f'E_{f.name}'] << u.MeV
            self.pinch[f] = simtab[f'ALPHA_{f.name}']
        super().__init__(time, metadata)


    def get_initial_spectra(self, t, E, flavors=Flavor):
        """Get neutrino spectra/luminosity curves before oscillation.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial spectra.
        flavors: iterable of snewpy.neutrino.Flavor
            Return spectra for these flavors only (default: all)

        Returns
        -------
        initialspectra : dict
            Dictionary of model spectra, keyed by neutrino flavor.
        """
        initialspectra = {}

        # Avoid division by zero in energy PDF below.
        E[E==0] = np.finfo(float).eps * E.unit

        # Estimate L(t), <E_nu(t)> and alpha(t). Express all energies in erg.
        E = E.to_value('erg')

        # Make sure input time uses the same units as the model time grid, or
        # the interpolation will not work correctly.
        t = t.to(self.time.unit)

        for flavor in flavors:
            # Use np.interp rather than scipy.interpolate.interp1d because it
            # can handle dimensional units (astropy.Quantity).
            L  = get_value(np.interp(t, self.time, self.luminosity[flavor].to('erg/s')))
            Ea = get_value(np.interp(t, self.time, self.meanE[flavor].to('erg')))
            a  = np.interp(t, self.time, self.pinch[flavor])

            # Sanity check to avoid invalid values of Ea, alpha, and L.
            initialspectra[flavor] = np.zeros_like(E, dtype=float) / (u.erg*u.s)
            if L <= 0. or Ea <= 0. or a <= -2.:
                continue
            # For numerical stability, evaluate log PDF and then exponentiate.
            initialspectra[flavor] = \
              np.exp(np.log(L) - (2+a)*np.log(Ea) + (1+a)*np.log(1+a)
                    - loggamma(1+a) + a*np.log(E) - (1+a)*(E/Ea)) / (u.erg * u.s)

        return initialspectra

    def get_transformed_spectra(self, t, E, flavor_xform, nudecay=False, rbar = 1.0, zeta = 1.0, model="phi0"):
        """Get neutrino spectra after applying oscillation.

        Parameters
        ----------
        t : astropy.Quantity
        Time to evaluate initial and oscillated spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
        Energies to evaluate the initial and oscillated spectra.
        flavor_xform : FlavorTransformation
        An instance from the flavor_transformation module.
        nudecay: bool (default: False)
        If true, implement energy-dependent neutrino decay
        rbar: float (default: 1.0)
        distance to supernova times decay width at 10MeV (dimensionless)
        zeta: float (default: 1.0)
        Visible neutrino fraction

        Returns
        -------
        dict
        Dictionary of transformed spectra, keyed by neutrino flavor.
        """
        transformed_spectra = super().get_transformed_spectra(t, E, flavor_xform)
        # initial_spectra = self.get_initial_spectra(t, E)
        if nudecay:
            eref = 10.0 * u.MeV
            eref_erg = eref.to('erg')
            ecoeff = eref_erg/u.erg

            # Neutrinos
            L_e  = get_value(np.interp(t, self.time, self.luminosity[Flavor.NU_E].to('erg/s')))
            Ea_e = get_value(np.interp(t, self.time, self.meanE[Flavor.NU_E].to('erg')))
            a_e  = np.interp(t, self.time, self.pinch[Flavor.NU_E])
            L_x  = get_value(np.interp(t, self.time, self.luminosity[Flavor.NU_X].to('erg/s')))
            Ea_x = get_value(np.interp(t, self.time, self.meanE[Flavor.NU_X].to('erg')))
            a_x  = np.interp(t, self.time, self.pinch[Flavor.NU_X])
            # Antineutrinos
            L_ebar  = get_value(np.interp(t, self.time, self.luminosity[Flavor.NU_E_BAR].to('erg/s')))
            Ea_ebar = get_value(np.interp(t, self.time, self.meanE[Flavor.NU_E_BAR].to('erg')))
            a_ebar  = np.interp(t, self.time, self.pinch[Flavor.NU_E_BAR])
            L_xbar  = get_value(np.interp(t, self.time, self.luminosity[Flavor.NU_X_BAR].to('erg/s')))
            Ea_xbar = get_value(np.interp(t, self.time, self.meanE[Flavor.NU_X_BAR].to('erg')))
            a_xbar  = np.interp(t, self.time, self.pinch[Flavor.NU_X_BAR])
            lumi = np.array([L_e, L_ebar, L_x, L_xbar])
            emean = np.array([Ea_e, Ea_ebar, Ea_x, Ea_xbar])/ecoeff
            alpha = np.array([a_e, a_ebar, a_x, a_xbar])

            heavy, medium, light, aheavy, amedium, alight = get_decayed_eigenstates(rbar, zeta, E/eref_erg, lumi, emean, alpha, ecoeff, model=model)

            # Get mixing parameters
            mass_ordering  = flavor_xform.mass_order
            theta12, theta13, theta23 = MixingParameters(mass_ordering).get_mixing_angles()
            Ue3_2 = np.sin(theta13)**2
            Ue2_2 = np.sin(theta12)**2 * np.cos(theta13)**2
            Ue1_2 = np.cos(theta12)**2 * np.cos(theta13)**2
            fe_final, fx_final, febar_final, fxbar_final = None, None, None, None
            if mass_ordering == MassHierarchy.NORMAL:
                fe_final = Ue3_2 * heavy + Ue2_2 * medium + Ue1_2 * light
                fx_final = (1-Ue3_2)/2 * heavy + (1-Ue2_2)/2 * medium + (1-Ue1_2)/2 * light
                febar_final = Ue3_2 * aheavy + Ue2_2 * amedium + Ue1_2 * alight
                fxbar_final = (1-Ue3_2)/2 * aheavy + (1-Ue2_2)/2 * amedium + (1-Ue1_2)/2 * alight
            else:
                fe_final = Ue2_2 * heavy + Ue1_2 * medium + Ue3_2 * light
                fx_final = (1-Ue2_2)/2 * heavy + (1-Ue1_2)/2 * medium + (1-Ue3_2)/2 * light
                febar_final = Ue2_2 * aheavy + Ue1_2 * amedium + Ue3_2 * alight
                fxbar_final = (1-Ue2_2)/2 * aheavy + (1-Ue1_2)/2 * amedium + (1-Ue3_2)/2 * alight

            transformed_spectra[Flavor.NU_E] = fe_final
            transformed_spectra[Flavor.NU_X] = fx_final
            transformed_spectra[Flavor.NU_E_BAR] = febar_final
            transformed_spectra[Flavor.NU_X_BAR] = fxbar_final

        return transformed_spectra   


class _GarchingArchiveModel(PinchedModel):
    """Subclass that reads models in the format used in the `Garching Supernova Archive <https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/>`_."""
    def __init__(self, filename, eos='LS220'):
        """Initialize model

        Parameters
        ----------
        filename : str
            Absolute or relative path to file prefix, we add nue/nuebar/nux.
        eos : string
            Equation of state used in simulation.
        """

        # Store model metadata.
        self.filename = os.path.basename(filename)
        self.EOS = eos
        self.progenitor_mass = float( (self.filename.split('s'))[1].split('c')[0] )  * u.Msun
        metadata = {
            'Progenitor mass':self.progenitor_mass,
            'EOS':self.EOS,
            }
        # Read through the several ASCII files for the chosen simulation and
        # merge the data into one giant table.
        mergtab = None
        for flavor in Flavor:
            _flav = Flavor.NU_X if flavor == Flavor.NU_X_BAR else flavor
            _sfx = _flav.name.replace('_', '').lower()
            _filename = '{}_{}_{}'.format(filename, eos, _sfx)
            _lname  = 'L_{}'.format(flavor.name)
            _ename  = 'E_{}'.format(flavor.name)
            _e2name = 'E2_{}'.format(flavor.name)
            _aname  = 'ALPHA_{}'.format(flavor.name)

            simtab = Table.read(_filename,
                                names=['TIME', _lname, _ename, _e2name],
                                format='ascii')
            simtab['TIME'].unit = 's'
            simtab[_lname].unit = '1e51 erg/s'
            simtab[_aname] = (2*simtab[_ename]**2 - simtab[_e2name]) / (simtab[_e2name] - simtab[_ename]**2)
            simtab[_ename].unit = 'MeV'
            del simtab[_e2name]

            if mergtab is None:
                mergtab = simtab
            else:
                mergtab = join(mergtab, simtab, keys='TIME', join_type='left')
                mergtab[_lname].fill_value = 0.
                mergtab[_ename].fill_value = 0.
                mergtab[_aname].fill_value = 0.
        simtab = mergtab.filled()
        super().__init__(simtab, metadata)

class _SegerlundModel(PinchedModel):
    """Subclass that reads models in the format used in the `Garching Supernova Archive <https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/>`_."""
    def __init__(self, filename, mass=27.0):
        """Initialize model

        Parameters
        ----------
        filename : str
            Absolute or relative path to file prefix, we add nue/nuebar/nux.
        mass : float
            Progenitor mass in units of Msun.
        """

        # Store model metadata.
        self.progenitor_mass = mass
        self.filename = os.path.basename(filename) + f"s{mass:.1f}_INS_small.dat"
        if not os.path.exists(filename + "/" + self.filename):
            self.filename = os.path.basename(filename) + f"s{int(mass)}_INS_small.dat"
        if not os.path.exists(filename + "/" + self.filename):
            self.filename = os.path.basename(filename) + f"s{mass:.2f}_INS_small.dat"
        _filename = filename + "/" +  self.filename
        metadata = {
            'Progenitor mass':self.progenitor_mass
            }
        # Read through the several ASCII files for the chosen simulation and
        # merge the data into one giant table.
        mergtab = None
        _lnames = []
        _enames = []
        _e2names = []
        _anames = []
        for flavor in Flavor:
            if flavor == Flavor.NU_X_BAR: continue
            _flav = flavor.name
            _lnames += ['L_{}'.format(_flav)]
            _enames  += ['E_{}'.format(_flav)]
            _e2names += ['E2_{}'.format(_flav)]
            _anames  += ['ALPHA_{}'.format(_flav)]
        simtab = Table.read(_filename,
                            names=['TIME', 'shock'] + _lnames + _enames + _e2names,
                            format='ascii')
        t_bounce = simtab['TIME'][simtab['shock']>0.00001][0]
        del simtab['shock']
        simtab['TIME'] -= t_bounce
        simtab['TIME'].unit = 's'

        simtab[f'L_{flavor.NU_X.name}'] /= 4 # Luminosity for one flavor
        simtab[f'L_{flavor.NU_X_BAR.name}'] = simtab[f'L_{flavor.NU_X.name}']
        simtab[f'E_{flavor.NU_X_BAR.name}'] = simtab[f'E_{flavor.NU_X.name}']
        simtab[f'E2_{flavor.NU_X_BAR.name}'] = simtab[f'E2_{flavor.NU_X.name}']
        _lnames.append(f'L_{flavor.NU_X_BAR.name}')
        _enames.append(f'E_{flavor.NU_X_BAR.name}')
        _e2names.append(f'E2_{flavor.NU_X_BAR.name}')
        _anames.append(f'ALPHA_{flavor.NU_X_BAR.name}')

        # Select line of sight
        maxrow = -1
        for _lname,_ename,_e2name,_aname in zip(_lnames,_enames,_e2names,_anames):
            simtab[_lname].unit = '1e51 erg/s'
            simtab[_aname] = (2*simtab[_ename]**2 - simtab[_e2name]**2) / (simtab[_e2name]**2 - simtab[_ename]**2)
            maxrow = max(np.where(simtab[_aname]>0)[0][0],maxrow)
            simtab[_ename].unit = 'MeV'
            del simtab[_e2name]

            if mergtab is None:
                mergtab = simtab
            mergtab[_lname].fill_value = 0.
            mergtab[_ename].fill_value = 0.
            mergtab[_aname].fill_value = 0.
        simtab = mergtab.filled()
        simtab = simtab[maxrow:]
        super().__init__(simtab, metadata)


