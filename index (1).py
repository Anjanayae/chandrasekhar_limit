import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Physical constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 2.99792458e8  # Speed of light (m/s)
hbar = 1.054571817e-34  # Reduced Planck constant (J⋅s)
m_e = 9.1093837015e-31  # Electron mass (kg)
m_p = 1.67262192369e-27  # Proton mass (kg)
M_sun = 1.989e30  # Solar mass (kg)

def degeneracy_pressure_nonrel(n_e):
    """Non-relativistic electron degeneracy pressure"""
    return (hbar**2 / (5 * m_e)) * (3 * np.pi**2)**(2/3) * n_e**(5/3)

def degeneracy_pressure_rel(n_e):
    """Ultra-relativistic electron degeneracy pressure"""
    return (hbar * c / 4) * (3 * np.pi**2)**(1/3) * n_e**(4/3)

def degeneracy_pressure_general(n_e):
    """General relativistic degeneracy pressure"""
    p_F = hbar * (3 * np.pi**2 * n_e)**(1/3)
    x = p_F / (m_e * c)
    
    factor = (m_e * c**2) / (8 * np.pi**2) * (hbar / (m_e * c))**(-3)
    f_x = x * (2*x**2 - 3) * np.sqrt(x**2 + 1) + 3 * np.arcsinh(x)
    
    return factor * f_x

def total_energy_density(rho, mu_e=2.0):
    """Total energy density including kinetic contributions"""
    n_e = rho / (mu_e * m_p)
    
    p_F = hbar * (3 * np.pi**2 * n_e)**(1/3)
    x = p_F / (m_e * c)
    
    kinetic_factor = (m_e * c**2) / (8 * np.pi**2) * (hbar / (m_e * c))**(-3)
    kinetic_integrand = x * (2*x**2 - 3) * np.sqrt(x**2 + 1) + 3 * np.arcsinh(x)
    kinetic_energy_density = kinetic_factor * kinetic_integrand
    
    return kinetic_energy_density

def total_energy(M, R, mu_e=2.0):
    """Total energy of white dwarf star"""
    rho = 3 * M / (4 * np.pi * R**3)
    
    kinetic_energy = 4 * np.pi * R**3 / 3 * total_energy_density(rho, mu_e)
    gravitational_energy = -3 * G * M**2 / (5 * R)
    
    return kinetic_energy + gravitational_energy

def find_equilibrium_radius(M, mu_e=2.0):
    """Find equilibrium radius by minimizing total energy"""
    def energy_func(R):
        if R <= 0:
            return np.inf
        return total_energy(M, R, mu_e)
    
    R_range = np.logspace(5, 7, 200)
    energies = [energy_func(R) for R in R_range]
    
    min_idx = np.argmin(energies)
    min_energy = energies[min_idx]
    
    if min_idx == 0 or min_idx == len(energies) - 1 or min_energy > 0:
        return None
    
    try:
        result = minimize_scalar(energy_func, bounds=(R_range[max(0, min_idx-5)], 
                                                    R_range[min(len(R_range)-1, min_idx+5)]), 
                               method='bounded')
        if result.success and result.fun < 0:
            return result.x
        else:
            return None
    except:
        return None

def chandrasekhar_mass_limit(mu_e=2.0):
    """Theoretical Chandrasekhar mass limit"""
    return 1.44 * M_sun * (mu_e / 2.0)**(-2)

# Pre-calculate stable mass-radius data for smooth animation
def precalculate_mass_radius_data():
    """Pre-calculate mass-radius data for different mu_e values"""
    mu_e_values = np.linspace(1.5, 3.0, 16)  # 16 points for smooth interpolation
    data = {}
    
    for mu_e in mu_e_values:
        M_ch = chandrasekhar_mass_limit(mu_e)
        masses = np.linspace(0.1, 0.98, 50) * M_ch  # Only up to 98% of M_ch
        
        stable_masses = []
        stable_radii = []
        
        for M in masses:
            R_eq = find_equilibrium_radius(M, mu_e)
            if R_eq is not None:
                stable_masses.append(M / M_sun)
                stable_radii.append(R_eq / 1e6)
        
        data[mu_e] = {'masses': stable_masses, 'radii': stable_radii, 'M_ch': M_ch/M_sun}
    
    return data, mu_e_values

# Pre-calculate data for smooth animations
print("Pre-calculating data for smooth animations...")
mass_radius_data, mu_e_grid = precalculate_mass_radius_data()
print("Data pre-calculation complete!")

# Create figure with better layout - increase figure height for more space
plt.style.use('default')
fig = plt.figure(figsize=(20, 16))

# Define layout with better spacing - adjusted for removed controls
gs = fig.add_gridspec(3, 2, 
                      height_ratios=[1, 1, 1], 
                      hspace=0.45, wspace=0.25, 
                      top=0.90, bottom=0.18, 
                      left=0.08, right=0.95)

# Create subplots
ax1 = fig.add_subplot(gs[0, 0])  # Mass vs Radius
ax2 = fig.add_subplot(gs[0, 1])  # Energy vs Radius
ax3 = fig.add_subplot(gs[1, 0])  # Pressure vs Density
ax4 = fig.add_subplot(gs[1, 1])  # Energy vs Mass
ax5 = fig.add_subplot(gs[2, :])  # Stability map

# Position sliders in the center of the page
slider_bottom = 0.08
slider_height = 0.015
slider_spacing = 0.025
slider_width = 0.30
slider_left = 0.35

ax_mu_e = plt.axes([slider_left, slider_bottom, slider_width, slider_height])
ax_current_mass = plt.axes([slider_left, slider_bottom - slider_spacing, slider_width, slider_height])
ax_density_scale = plt.axes([slider_left, slider_bottom - 2*slider_spacing, slider_width, slider_height])

# Create sliders
mu_e_slider = Slider(ax_mu_e, 'μₑ', 1.5, 3.0, valinit=2.0, valfmt='%.2f')
current_mass_slider = Slider(ax_current_mass, 'Mass (M☉)', 0.5, 1.8, valinit=1.0, valfmt='%.2f')
density_scale_slider = Slider(ax_density_scale, 'Density Scale', 0.1, 5.0, valinit=1.0, valfmt='%.1f')

# Store line objects for smooth animation
line_objects = {}

def interpolate_mass_radius_data(mu_e_target):
    """Interpolate mass-radius data for smooth animation"""
    if mu_e_target in mass_radius_data:
        return mass_radius_data[mu_e_target]
    
    # Find surrounding values for interpolation
    mu_e_lower = max([mu for mu in mu_e_grid if mu <= mu_e_target])
    mu_e_upper = min([mu for mu in mu_e_grid if mu >= mu_e_target])
    
    if mu_e_lower == mu_e_upper:
        return mass_radius_data[mu_e_lower]
    
    # Linear interpolation
    alpha = (mu_e_target - mu_e_lower) / (mu_e_upper - mu_e_lower)
    
    data_lower = mass_radius_data[mu_e_lower]
    data_upper = mass_radius_data[mu_e_upper]
    
    # Interpolate M_ch
    M_ch_interp = data_lower['M_ch'] * (1 - alpha) + data_upper['M_ch'] * alpha
    
    # For simplicity, use the closest data set for masses/radii
    if alpha < 0.5:
        return {'masses': data_lower['masses'], 'radii': data_lower['radii'], 'M_ch': M_ch_interp}
    else:
        return {'masses': data_upper['masses'], 'radii': data_upper['radii'], 'M_ch': M_ch_interp}

def init_plots():
    """Initialize all plots with empty data"""
    # Initialize line objects for smooth updates
    line_objects['mass_radius'], = ax1.plot([], [], 'b-', linewidth=3, label='Stable White Dwarfs')
    line_objects['ch_limit_1'] = ax1.axhline(y=1.44, color='red', linestyle='--', linewidth=3, label='Chandrasekhar Limit')
    line_objects['current_mass_1'] = ax1.axhline(y=1.0, color='green', linestyle=':', alpha=0.7, label='Current mass')
    
    # Energy plots
    line_objects['energy_curves'] = []
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    labels = ['0.6 M_ch', '0.8 M_ch', '1.0 M_ch', '1.2 M_ch', '1.4 M_ch']
    for i, (color, label) in enumerate(zip(colors, labels)):
        line, = ax2.plot([], [], color=color, linewidth=2, label=label)
        line_objects['energy_curves'].append(line)
    
    # Pressure plots
    line_objects['pressure_nonrel'], = ax3.loglog([], [], 'b--', label='Non-relativistic', linewidth=2)
    line_objects['pressure_rel'], = ax3.loglog([], [], 'r--', label='Ultra-relativistic', linewidth=2)
    line_objects['pressure_general'], = ax3.loglog([], [], 'k-', label='General relativistic', linewidth=3)
    line_objects['current_density'] = ax3.axvline(x=1, color='green', linestyle=':', alpha=0.7, label='Current density')
    
    # Energy vs mass plot
    line_objects['min_energy'], = ax4.plot([], [], 'b-', linewidth=3, label='Minimum Energy')
    line_objects['instability'] = ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Instability threshold')
    line_objects['ch_limit_4'] = ax4.axvline(x=1.44, color='red', linestyle='--', linewidth=2, alpha=0.7)
    line_objects['current_mass_4'] = ax4.axvline(x=1.0, color='green', linestyle=':', alpha=0.7, label='Current mass')

def update_plots(val=None):
    """Update all plots efficiently"""
    # Get slider values
    mu_e = mu_e_slider.val
    current_mass = current_mass_slider.val
    density_scale = density_scale_slider.val
    
    # Get interpolated data
    data = interpolate_mass_radius_data(mu_e)
    M_ch = data['M_ch']
    
    # Update Plot 1: Mass vs Radius with sharp cutoff
    if data['masses'] and data['radii']:
        line_objects['mass_radius'].set_data(data['radii'], data['masses'])
    
    line_objects['ch_limit_1'].set_ydata([M_ch, M_ch])
    line_objects['current_mass_1'].set_ydata([current_mass, current_mass])
    
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 2.5)
    ax1.set_xlabel('Radius (1000 km)', fontsize=10)
    ax1.set_ylabel('Mass (M☉)', fontsize=10)
    ax1.set_title('Mass vs Radius: Sharp Cutoff', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Update Plot 2: Energy vs Radius
    R_range = np.linspace(1e6, 1e7, 50)
    M_ch_kg = M_ch * M_sun
    test_masses_frac = [0.6, 0.8, 1.0, 1.2, 1.4]
    
    for i, frac in enumerate(test_masses_frac):
        M_test = frac * M_ch_kg
        energies = [total_energy(M_test, R, mu_e) / (M_test * c**2) for R in R_range]
        line_objects['energy_curves'][i].set_data(R_range/1e6, energies)
    
    ax2.set_xlim(1, 10)
    ax2.set_ylim(-0.0003, 0.0003)
    ax2.set_xlabel('Radius (1000 km)', fontsize=10)
    ax2.set_ylabel('Total Energy / (Mc²)', fontsize=10)
    ax2.set_title('Energy vs Radius', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Update Plot 3: Pressure vs Density
    densities = np.logspace(8, 12, 50) * density_scale
    n_e_array = densities / (mu_e * m_p)
    
    P_nonrel = [degeneracy_pressure_nonrel(n) for n in n_e_array]
    P_rel = [degeneracy_pressure_rel(n) for n in n_e_array]
    P_general = [degeneracy_pressure_general(n) for n in n_e_array]
    
    line_objects['pressure_nonrel'].set_data(densities/1e9, np.array(P_nonrel)/1e15)
    line_objects['pressure_rel'].set_data(densities/1e9, np.array(P_rel)/1e15)
    line_objects['pressure_general'].set_data(densities/1e9, np.array(P_general)/1e15)
    
    current_rho = 3 * current_mass * M_sun / (4 * np.pi * (5e6)**3)
    line_objects['current_density'].set_xdata([current_rho/1e9, current_rho/1e9])
    
    ax3.set_xlim(1e-1, 1e3)
    ax3.set_ylim(1e4, 1e12)
    ax3.set_xlabel('Density (10⁹ kg/m³)', fontsize=10)
    ax3.set_ylabel('Pressure (10¹⁵ Pa)', fontsize=10)
    ax3.set_title('Equation of State', fontsize=11, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Update Plot 4: Energy vs Mass
    mass_range = np.linspace(0.5, M_ch * 0.99, 30)
    min_energies = []
    
    for M in mass_range:
        R_eq = find_equilibrium_radius(M * M_sun, mu_e)
        if R_eq is not None:
            E_min = total_energy(M * M_sun, R_eq, mu_e) / (M * M_sun * c**2)
            min_energies.append(E_min)
        else:
            min_energies.append(np.nan)
    
    valid_mask = ~np.isnan(min_energies)
    if np.any(valid_mask):
        line_objects['min_energy'].set_data(mass_range[valid_mask], np.array(min_energies)[valid_mask])
    
    line_objects['ch_limit_4'].set_xdata([M_ch, M_ch])
    line_objects['current_mass_4'].set_xdata([current_mass, current_mass])
    
    ax4.set_xlim(0.5, 2.0)
    ax4.set_ylim(-1e-5, 1e-6)
    ax4.set_xlabel('Mass (M☉)', fontsize=10)
    ax4.set_ylabel('Minimum Energy / (Mc²)', fontsize=10)
    ax4.set_title('Energy vs Mass', fontsize=11, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Update Plot 5: Stability Map
    ax5.clear()
    
    mass_range_map = np.linspace(0.3, 2.5, 40)
    radius_range_map = np.linspace(1, 12, 40)
    
    M_mesh, R_mesh = np.meshgrid(mass_range_map, radius_range_map)
    stability = np.zeros_like(M_mesh)
    
    for i in range(len(mass_range_map)):
        for j in range(len(radius_range_map)):
            if mass_range_map[i] < M_ch:
                stability[j, i] = 1  # Stable
            else:
                stability[j, i] = 0  # Unstable
    
    # Create contour
    levels = [0, 0.5, 1]
    colors = ['red', 'lightblue']
    ax5.contourf(M_mesh, R_mesh, stability, levels=levels, colors=colors, alpha=0.6)
    
    # Add theoretical curve
    if data['masses'] and data['radii']:
        ax5.plot(data['masses'], data['radii'], 'k-', linewidth=4, label='Theoretical M-R curve')
    
    # Mark current point
    current_R = find_equilibrium_radius(current_mass * M_sun, mu_e)
    if current_R is not None:
        ax5.plot(current_mass, current_R/1e6, 'go', markersize=10, 
                label='Current config', markeredgecolor='black', markeredgewidth=2)
    
    ax5.axvline(x=M_ch, color='red', linestyle='--', linewidth=3, 
                label=f'M_Ch = {M_ch:.2f} M☉')
    
    ax5.set_xlabel('Mass (M☉)', fontsize=10)
    ax5.set_ylabel('Radius (1000 km)', fontsize=10)
    ax5.set_title('White Dwarf Stability Map', fontsize=11, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0.3, 2.5)
    ax5.set_ylim(1, 12)
    
    # Add region labels with smaller font
    ax5.text(0.8, 10, 'STABLE\nWHITE DWARFS', ha='center', va='center', 
             fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    ax5.text(2.1, 10, 'UNSTABLE\n(COLLAPSE)', ha='center', va='center', 
             fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.9))
    
    # Force redraw
    fig.canvas.draw_idle()

# Initialize plots
init_plots()

# Connect sliders
mu_e_slider.on_changed(update_plots)
current_mass_slider.on_changed(update_plots)
density_scale_slider.on_changed(update_plots)

# Initial update
update_plots()

# Add title with better positioning - moved up more
fig.suptitle('Chandrasekhar Limit: Sharp Cutoff in White Dwarf Physics', 
             fontsize=14, fontweight='bold', y=0.96)

plt.tight_layout()
plt.show()

# Print final results
print(f"\nChandrasekhar Mass (μₑ=2.0): {chandrasekhar_mass_limit(2.0)/M_sun:.3f} M☉")
print("Sharp cutoff demonstrates the fundamental limit of white dwarf stability!")