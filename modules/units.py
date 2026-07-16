#> name: units.py
#> author: John Miller Jr
#> descrp: all relevant units for calculations

""" #> UNITS ==========================
================================== """

class u():

    #> class params
    def __init__(self):
        
        #> unit conversions
        self.cm_kpc = 3.08567758128e+21   # cm / kpc
        self.cm_mpc = 3.08567758128e+24   # cm / Mpc
        self.km_mpc = 3.08567758128e+19   # km / Mpc
        
        self.arc_rad = 206264.8062471  # arcsec/rad
        
        #> speed of light
        self.c_km = 2.99792e+5         # km/s
        self.c_m = self.c_km * 1000    # m/s
        self.c_cm = self.c_m * 100     # cm/s
        
        #> physical units
        self.solMass = 1.989e+33 # g
        
        #> constants
        self.G = 6.67259e-8  # cm^3/g/s^2
        self.G_kpc_solMass = self.G / self.cm_kpc**3 * self.solMass # kpc^3/solMass/s^2
        
        #> standard cosmology
        self.cosmo = {'h0': 70.000, 
                     'omega_m': 0.27,
                     'omega_lam': 0.73
                     }
        
        #> defaults for grid
        self.pix_arc = 60
        self.nph = 50
        
        
""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    # end
# thank