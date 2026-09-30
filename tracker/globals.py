class constants:
    L_Al =  0.4                             # [cm] Aluminum
    L_Sc = 1.0                              # [cm] Scintillator
    L_r_Al = 24.0111/2.7                    # [cm] Radiation length Aluminum/ density of Aluminum
    L_r_Sc = 43                             # [cm] Radiation length Scintillator (Saint-Gobain paper)
    L_rad = L_Al / L_r_Al + L_Sc / L_r_Sc   # [rad lengths] orthogonal to Layer