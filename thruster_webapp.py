import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import math

# Configure page
st.set_page_config(
    page_title="Butane Thruster Simulation", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Enhanced CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        --secondary-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --accent-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --glass-bg: rgba(255, 255, 255, 0.25);
        --glass-border: rgba(255, 255, 255, 0.18);
        --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.37);
        --border-radius: 20px;
    }
    
    .main .block-container {
        padding-top: 1rem;
        max-width: 1600px;
        margin: 0 auto;
    }
    
    .hero-header {
        background: var(--primary-gradient);
        padding: 3rem 2rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-light);
        text-align: center;
        color: white;
    }
    
    .hero-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        text-shadow: 2px 2px 20px rgba(0,0,0,0.5);
    }
    
    .formula-display {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 2px solid #0f3460;
        border-radius: 15px;
        padding: 1.5rem;
        color: #e94560;
        font-family: 'JetBrains Mono', monospace;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.1rem;
    }
    
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
    }
    
    .validation-excellent { 
        color: #00ff88; 
        font-weight: 700; 
        background: rgba(0, 255, 136, 0.1);
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
    }
    
    .validation-good { 
        color: #28a745; 
        font-weight: 700;
        background: rgba(40, 167, 69, 0.1);
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    .validation-warning { 
        color: #ffc107; 
        font-weight: 700;
        background: rgba(255, 193, 7, 0.1);
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    
    .validation-danger {
        color: #dc3545;
        font-weight: 700;
        background: rgba(220, 53, 69, 0.1);
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
    
    .thermodynamic-box {
        background: var(--secondary-gradient);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
    }
    
    .parameter-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .performance-summary {
        background: var(--accent-gradient);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
    }
    
    .nozzle-info {
        background: rgba(103, 126, 234, 0.1);
        border: 1px solid rgba(103, 126, 234, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Physical Constants for Butane
GAMMA = 1.05  # Specific heat ratio for butane (k)
R_BUTANE = 143.05  # Gas constant for butane (J/kg·K)
G0 = 9.80665  # Standard gravity (m/s²)
MOLECULAR_WEIGHT_BUTANE = 58.12  # g/mol

# Enhanced Cp data for butane (temperature in K, Cp in J/kg·K)
# Based on NIST data with extended temperature range
T_CP_DATA = np.array([200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 
                      750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 
                      1500, 1600, 1700, 1800, 1900, 2000])
CP_DATA = np.array([1690, 1790, 1880, 1960, 2040, 2110, 2170, 2230, 2280, 2330, 2380,
                    2420, 2460, 2500, 2530, 2560, 2590, 2640, 2680, 2720, 2750, 
                    2780, 2800, 2820, 2840, 2860, 2880])

# Create interpolation function for Cp(T)
cp_interp = interp1d(T_CP_DATA, CP_DATA, kind='cubic', fill_value='extrapolate')

def get_cp(T):
    """Get specific heat capacity at temperature T using interpolation."""
    return float(cp_interp(T))

def compute_ideal_exhaust_velocity(T_chamber, P_chamber, P_exit, gamma=GAMMA, R=R_BUTANE):
    """
    Compute ideal exhaust velocity using the specification formula:
    υ₂ = √[(2k)/(k-1) * R*T₁ * (1 - (P₂/P₁)^((k-1)/k)]
    
    Args:
        T_chamber: Chamber temperature T₁ (K)
        P_chamber: Chamber pressure P₁ (Pa)
        P_exit: Exit pressure P₂ (Pa)
        gamma: Specific heat ratio k
        R: Gas constant (J/kg·K)
    
    Returns:
        Ideal exhaust velocity (m/s)
    """
    if P_chamber <= P_exit:
        return 0.0
    
    pressure_ratio = P_exit / P_chamber
    pressure_ratio = max(pressure_ratio, 1e-10)  # Prevent numerical issues
    
    # Formula from specification: υ₂ = √[(2k)/(k-1) * R*T₁ * (1 - (P₂/P₁)^((k-1)/k)]
    term1 = (2 * gamma) / (gamma - 1)
    term2 = R * T_chamber
    term3 = 1 - pressure_ratio**((gamma - 1) / gamma)
    
    if term3 < 0:
        term3 = 0
    
    v_ideal = np.sqrt(term1 * term2 * term3)
    return v_ideal

def compute_orifice_area_from_specification(mass_flow_rate, P_chamber, T_chamber, gamma=GAMMA, R=R_BUTANE):
    """
    Compute orifice area using the formula:
    A_orifice = (ṁ/P₁) * √[ (R*T₁) / (k * [2/(k+1)]^((k+1)/(k-1))) ]
    
    Args:
        mass_flow_rate: Mass flow rate ṁ (kg/s)
        P_chamber: Chamber pressure P₁ (Pa)
        T_chamber: Chamber temperature T₁ (K)
        gamma: Specific heat ratio k
        R: Gas constant (J/kg·K)
    
    Returns:
        Orifice area (m²)
    """
    if mass_flow_rate <= 0 or P_chamber <= 0 or T_chamber <= 0:
        return 0.0
    
    # Formula: A_orifice = (ṁ/P₁) * √[ (R*T₁) / (k * [2/(k+1)]^((k+1)/(k-1))) ]
    term1 = mass_flow_rate / P_chamber
    term2_numerator = R * T_chamber
    term2_denominator = gamma * ((2 / (gamma + 1))**((gamma + 1) / (gamma - 1)))
    term2 = np.sqrt(term2_numerator / term2_denominator)
    
    A_orifice = term1 * term2
    return A_orifice

def compute_exit_area_from_specification(A_orifice, P_chamber, P_exit, gamma=GAMMA):
    """
    Compute exit area using the formula:
    A₂ = A_orifice * [ ( (k+1)/2 )^(1/(k-1)) * (P₂/P₁)^(1/k) * √( (k+1)/(k-1) * (1 - (P₂/P₁)^((k-1)/k)) ) ]
    
    Args:
        A_orifice: Orifice area (m²)
        P_chamber: Chamber pressure P₁ (Pa)
        P_exit: Exit pressure P₂ (Pa)
        gamma: Specific heat ratio k
    
    Returns:
        Exit area (m²)
    """
    if P_chamber <= P_exit or A_orifice <= 0:
        return A_orifice
    
    pressure_ratio = P_exit / P_chamber
    pressure_ratio = max(pressure_ratio, 1e-10)
    
    # Formula: A₂ = A_orifice * [ ( (k+1)/2 )^(1/(k-1)) * (P₂/P₁)^(1/k) * √( (k+1)/(k-1) * (1 - (P₂/P₁)^((k-1)/k)) ) ]
    term1 = ((gamma + 1) / 2)**(1 / (gamma - 1))
    term2 = pressure_ratio**(1 / gamma)
    term3_inner = ((gamma + 1) / (gamma - 1)) * (1 - pressure_ratio**((gamma - 1) / gamma))
    term3_inner = max(term3_inner, 0)  # Prevent negative values under square root
    term3 = np.sqrt(term3_inner)
    
    A_exit = A_orifice * term1 * term2 * term3
    return A_exit

def compute_thrust_force_from_specification(mass_flow_rate, exhaust_velocity, P_exit, P_ambient, A_exit):
    """
    Compute thrust force using the specification formula:
    F = ṁ*υ₂ + (P₂ - P₃)*A₂
    
    Args:
        mass_flow_rate: Mass flow rate ṁ (kg/s)
        exhaust_velocity: Exhaust velocity υ₂ (m/s)
        P_exit: Nozzle exit pressure P₂ (Pa)
        P_ambient: Ambient pressure P₃ (Pa)
        A_exit: Exit area A₂ (m²)
    
    Returns:
        Thrust force F (N), momentum thrust (N), pressure thrust (N)
    """
    momentum_thrust = mass_flow_rate * exhaust_velocity
    pressure_thrust = (P_exit - P_ambient) * A_exit
    total_thrust = momentum_thrust + pressure_thrust
    
    return total_thrust, momentum_thrust, pressure_thrust

def compute_exit_pressure_from_expansion_ratio(P_chamber, expansion_ratio, gamma=GAMMA):
    """
    Computes the nozzle exit pressure for a given expansion ratio and chamber pressure.
    Uses a numerical solver to find P_exit.
    
    Args:
        P_chamber: Chamber pressure (Pa)
        expansion_ratio: Nozzle expansion ratio (A_e / A_orifice)
        gamma: Specific heat ratio
        
    Returns:
        Exit pressure (Pa)
    """
    if expansion_ratio <= 1 or P_chamber <= 0:
        return P_chamber
        
    def pressure_ratio_residual(pressure_ratio):
        """
        The function to find the root of. We want to find the pressure_ratio
        that makes this function equal to zero.
        """
        # Isentropic flow equation for area ratio
        # A_exit / A_orifice = ( (gamma+1)/2 )^(1/(gamma-1)) * (P_e/P_c)^(1/gamma) * sqrt( (gamma+1)/(gamma-1) * (1 - (P_e/P_c)^((gamma-1)/gamma)) )
        term1 = ((gamma + 1) / 2)**(1 / (gamma - 1))
        term2 = pressure_ratio**(1 / gamma)
        term3_inner = ((gamma + 1) / (gamma - 1)) * (1 - pressure_ratio**((gamma - 1) / gamma))
        term3_inner = max(term3_inner, 1e-10) # Avoid negative values under the sqrt
        
        calculated_expansion_ratio = term1 * term2 * np.sqrt(term3_inner)
        
        return calculated_expansion_ratio - expansion_ratio
        
    initial_guess_pr = 0.1
    pr_found = fsolve(pressure_ratio_residual, initial_guess_pr) 
    
    # The pressure ratio (P_exit / P_chamber) cannot be greater than 1
    if pr_found > 1:
        pr_found = 1
    
    P_exit_found = P_chamber * pr_found[0]
    return P_exit_found

def compute_specific_impulse(exhaust_velocity):
    """
    Compute specific impulse from exhaust velocity.
    Isp = υ₂ / g₀
    """
    return exhaust_velocity / G0

def update_chamber_temperature_dynamic_cp(T_current, Q_input, mass_flow_rate, dt):
    if mass_flow_rate <= 0:
        return T_current
    
    # Get dynamic Cp at current temperature
    cp_current = get_cp(T_current)
    
    # Prevent division by zero or unrealistic values
    if cp_current < 1e-6:
        return T_current
    
    # Energy balance: Q = ṁ * Cp * dT/dt → dT/dt = Q / (ṁ * Cp)
    dT_dt = Q_input / (mass_flow_rate * cp_current)
    T_new = T_current + dT_dt * dt
    
    # Ensure temperature stays within reasonable bounds
    T_new = max(T_new, 200)  # Minimum reasonable temperature
    T_new = min(T_new, 3000)  # Maximum reasonable temperature
    
    return T_new

def compute_chamber_pressure_from_choked_flow(T_chamber, mass_flow_rate, A_orifice, gamma=GAMMA, R=R_BUTANE):
    if T_chamber <= 0 or mass_flow_rate <= 0 or A_orifice <= 0:
        return 0.0
    
    # Choked flow mass flow rate: ṁ = A* * P₁ * √(k/RT₁) * (2/(k+1))^((k+1)/(2(k-1)))
    # Solving for P₁: P₁ = ṁ / (A* * √(k/RT₁) * (2/(k+1))^((k+1)/(2(k-1))))
    
    term1 = np.sqrt(gamma / (R * T_chamber))
    term2 = ((2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1))))
    choked_coeff = term1 * term2
    
    P_chamber = mass_flow_rate / (A_orifice * choked_coeff)
    return P_chamber

def compute_orifice_mass_flow_rate(orifice_area, upstream_pressure, downstream_pressure, temperature, 
                                 discharge_coefficient=0.6, gamma=GAMMA, R=R_BUTANE):
    if upstream_pressure <= downstream_pressure or orifice_area <= 0 or temperature <= 0:
        return 0.0
    
    pressure_ratio = downstream_pressure / upstream_pressure
    critical_pressure_ratio = (2 / (gamma + 1))**(gamma / (gamma - 1))
    
    # Density at upstream conditions
    rho_upstream = upstream_pressure / (R * temperature)
    
    if pressure_ratio <= critical_pressure_ratio:
        # Choked flow
        velocity = np.sqrt(gamma * R * temperature)  # Sonic velocity
        mass_flow_rate = discharge_coefficient * orifice_area * rho_upstream * velocity * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))
    else:
        # Unchoked flow
        term1 = 2 * gamma / (gamma - 1)
        term2 = pressure_ratio**(2 / gamma) - pressure_ratio**((gamma + 1) / gamma)
        velocity = np.sqrt(term1 * R * temperature * term2)
        rho_downstream = downstream_pressure / (R * temperature)
        mass_flow_rate = discharge_coefficient * orifice_area * np.sqrt(2 * rho_upstream * (upstream_pressure - downstream_pressure))
    
    return mass_flow_rate

def validate_simulation_parameters(params):
    """
    Validate simulation parameters and return warnings/errors.
    """
    messages = []
    
    # Temperature validation
    if params['T_initial'] < 200:
        messages.append(("danger", "Initial chamber temperature too low (< 200K). May cause unrealistic results."))
    elif params['T_initial'] > 1500:
        messages.append(("warning", "Initial chamber temperature very high (> 1500K). Ensure realistic for your application."))
    elif params['T_initial'] > 500:
        messages.append(("good", "Initial chamber temperature in good range for resistojet operation."))
    else:
        messages.append(("excellent", "Initial chamber temperature in excellent range for cold gas thrusters."))
    
    # Expansion ratio validation
    if params['expansion_ratio'] < 1:
        messages.append(("danger", "Expansion ratio must be ≥ 1."))
    elif params['expansion_ratio'] > 50:
        messages.append(("warning", "Very high expansion ratio. Ensure nozzle design is practical."))
    elif params['expansion_ratio'] > 10:
        messages.append(("good", "Good expansion ratio for efficient nozzle operation."))
    else:
        messages.append(("excellent", "Excellent expansion ratio for compact thruster design."))
    
    # Orifice validation
    if params['orifice_radius'] <= 0:
        messages.append(("danger", "Orifice radius must be positive."))
    
    # Flow validation
    expected_flow = params['orifice_radius'] * 1000  # Rough estimate
    if expected_flow < 0.001:
        messages.append(("warning", "Very small orifice may result in very low flow rates."))
    elif expected_flow > 1.0:
        messages.append(("warning", "Large orifice may result in very high flow rates."))
    else:
        messages.append(("good", "Orifice size appears reasonable."))
    
    return messages

def run_enhanced_simulation(params):
    """
    Run the enhanced thruster simulation following project specifications.
    """
    # Extract parameters
    T_initial = params['T_initial']
    Q_input = params['Q_input']
    expansion_ratio = params['expansion_ratio']
    P_ambient = params['P_ambient']
    total_time = params['total_time']
    dt = params['dt']
    
    # Tank and orifice parameters
    initial_butane_mass = params['initial_butane_mass']
    tank_volume = params['tank_volume']  # L
    tank_temperature = params['tank_temperature']  # K
    discharge_coefficient = params['discharge_coefficient']
    orifice_radius = params['orifice_radius']
    
    # Calculate fixed areas from user inputs
    A_orifice = np.pi * (orifice_radius * 1e-3)**2
    A_exit_fixed = A_orifice * expansion_ratio  # Exit area based on expansion ratio
    
    # Calculate exit radius for display
    exit_radius = np.sqrt(A_exit_fixed / np.pi) * 1e3  # Convert to mm
    
    # Nozzle efficiency
    nozzle_efficiency = params['nozzle_efficiency'] / 100
    
    # Initialize arrays
    times = np.arange(0, total_time + dt, dt)
    n_steps = len(times)
    
    # Result arrays
    temperatures = np.zeros(n_steps)
    chamber_pressures = np.zeros(n_steps)
    exit_pressures = np.zeros(n_steps)
    exhaust_velocities = np.zeros(n_steps)
    specific_impulses = np.zeros(n_steps)
    thrust_forces = np.zeros(n_steps)
    momentum_thrusts = np.zeros(n_steps)
    pressure_thrusts = np.zeros(n_steps)
    cp_values = np.zeros(n_steps)
    mass_flow_rates = np.zeros(n_steps)
    tank_pressures = np.zeros(n_steps)
    butane_masses = np.zeros(n_steps)
    orifice_areas_computed = np.zeros(n_steps)
    exit_areas_computed = np.zeros(n_steps)
    
    # Initial conditions
    T_chamber = T_initial
    current_butane_mass = initial_butane_mass
    
    # Simulation loop
    for i in range(n_steps):
        if current_butane_mass > 0:
            tank_pressure_pa = (current_butane_mass / (tank_volume * 1e-3)) * R_BUTANE * tank_temperature
        else:
            tank_pressure_pa = 0.0
        
        tank_pressures[i] = tank_pressure_pa
        butane_masses[i] = current_butane_mass
        temperatures[i] = T_chamber
        cp_values[i] = get_cp(T_chamber)
        
        # Compute mass flow rate through orifice
        P_chamber_estimate = tank_pressure_pa * 0.8  
        
        mass_flow_rate = compute_orifice_mass_flow_rate(
            A_orifice, tank_pressure_pa, P_chamber_estimate, 
            tank_temperature, discharge_coefficient, GAMMA, R_BUTANE
        )
        
        # Limit mass flow rate to available mass
        mass_flow_rate = min(mass_flow_rate, current_butane_mass / dt)
        mass_flow_rates[i] = mass_flow_rate
        
        if mass_flow_rate <= 0:
            # No flow
            chamber_pressures[i] = 0.0
            exit_pressures[i] = 0.0
            exhaust_velocities[i] = 0.0
            specific_impulses[i] = 0.0
            thrust_forces[i] = 0.0
            momentum_thrusts[i] = 0.0
            pressure_thrusts[i] = 0.0
            orifice_areas_computed[i] = 0.0
            exit_areas_computed[i] = 0.0
            continue
        
        P_chamber = compute_chamber_pressure_from_choked_flow(T_chamber, mass_flow_rate, A_orifice, GAMMA, R_BUTANE)
        chamber_pressures[i] = P_chamber
    
        # Calculate exit pressure based on chamber pressure and expansion ratio
        P_exit = compute_exit_pressure_from_expansion_ratio(P_chamber, expansion_ratio, GAMMA)
        exit_pressures[i] = P_exit

        # Compute areas using specification formulas (for comparison/validation)
        A_orifice_spec = compute_orifice_area_from_specification(mass_flow_rate, P_chamber, T_chamber, GAMMA, R_BUTANE)
        A_exit_spec = compute_exit_area_from_specification(A_orifice, P_chamber, P_exit, GAMMA)
        orifice_areas_computed[i] = A_orifice_spec
        exit_areas_computed[i] = A_exit_spec
        
        # Compute ideal exhaust velocity using specification formula
        v_ideal = compute_ideal_exhaust_velocity(T_chamber, P_chamber, P_exit, GAMMA, R_BUTANE)
        v_actual = v_ideal * nozzle_efficiency  # Apply efficiency
        exhaust_velocities[i] = v_actual
        
        # Compute specific impulse
        isp = compute_specific_impulse(v_actual)
        specific_impulses[i] = isp
        
        # Compute thrust using specification formula
        F_total, F_momentum, F_pressure = compute_thrust_force_from_specification(
            mass_flow_rate, v_actual, P_exit, P_ambient, A_exit_fixed
        )
        thrust_forces[i] = F_total
        momentum_thrusts[i] = F_momentum
        pressure_thrusts[i] = F_pressure
        
        # Update temperature for next time step using dynamic Cp
        if i < n_steps - 1:
            T_chamber = update_chamber_temperature_dynamic_cp(T_chamber, Q_input, mass_flow_rate, dt)
            current_butane_mass -= mass_flow_rate * dt
            current_butane_mass = max(current_butane_mass, 0)
    
    # Return the final results after the loop finishes
    return {
        'times': times,
        'temperatures': temperatures,
        'chamber_pressures': chamber_pressures,
        'exit_pressures': exit_pressures,
        'exhaust_velocities': exhaust_velocities,
        'specific_impulses': specific_impulses,
        'thrust_forces': thrust_forces,
        'momentum_thrusts': momentum_thrusts,
        'pressure_thrusts': pressure_thrusts,
        'cp_values': cp_values,
        'mass_flow_rates': mass_flow_rates,
        'tank_pressures': tank_pressures,
        'butane_masses': butane_masses,
        'orifice_areas_computed': orifice_areas_computed,
        'exit_areas_computed': exit_areas_computed,
        'nozzle_geometry': {
            'orifice_area': A_orifice,
            'exit_area_fixed': A_exit_fixed,
            'expansion_ratio': expansion_ratio,
            'exit_radius_mm': exit_radius
        },
        'final_values': {
            'temperature': temperatures[-1],
            'chamber_pressure': chamber_pressures[-1],
            'exit_pressure': exit_pressures[-1],
            'exhaust_velocity': exhaust_velocities[-1],
            'specific_impulse': specific_impulses[-1],
            'thrust': thrust_forces[-1],
            'cp': cp_values[-1],
            'mass_flow_rate': mass_flow_rates[-1],
            'tank_pressure': tank_pressures[-1]
        }
    }

# Streamlit Application
def main():
    # Header
    st.markdown("""
    <div class="hero-header">
        <h1>🚀 Butane Thruster Simulation</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Complete thermodynamic modeling with tunable nozzle geometry and tank parameters
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input parameters
    with st.sidebar:
        st.header("🔧 Simulation Parameters")
    
        # --- Butane Tank Settings ---
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.subheader("Butane Tank Settings")

        tank_volume = st.number_input(
            "Tank Volume (L)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.2f",
            help="Volume of the butane tank in Liters."
        )

        tank_temperature = st.number_input(
            "Tank Temperature (K)",
            min_value=200.0,
            max_value=400.0,
            value=293.15,
            step=0.1,
            format="%.2f",
            help="Temperature of the butane tank in Kelvin."
        )

        initial_butane_mass = st.number_input(
            "Initial Butane Mass (kg)",
            min_value=0.001,
            max_value=10.0,
            value=0.100,
            step=0.01,
            format="%.3f",
            help="Initial mass of butane in the tank."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Thruster Orifice and Nozzle Settings ---
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.subheader("Orifice & Nozzle Design")

        # Orifice (flow choking point) radius
        orifice_radius = st.number_input(
            "Orifice Radius (mm)",
            min_value=0.025,
            max_value=2.0,
            value=0.125,
            step=0.005,
            format="%.3f",
            help="The radius of the flow choking point (orifice). This is the critical area that determines mass flow rate."
        )

        discharge_coefficient = st.number_input(
            "Discharge Coefficient",
            min_value=0.5,
            max_value=1.0,
            value=0.6,
            step=0.01,
            format="%.2f",
            help="Coefficient of discharge for the orifice flow."
        )

        # --- Nozzle Design Tool Logic ---
        st.subheader("Nozzle Expansion Design")
        nozzle_mode = st.radio(
            "Configure Nozzle by:",
            ("Set Expansion Ratio", "Set Exit Radius"),
            index=0,  # Default to expansion ratio
            help="Choose whether to define the nozzle geometry by its expansion ratio or exit radius."
        )
        
        # Initialize session state
        if 'exit_radius' not in st.session_state:
            st.session_state['exit_radius'] = 0.5
        if 'expansion_ratio' not in st.session_state:
            st.session_state['expansion_ratio'] = 16.0

        # Initialize the variables for the main scope
        exit_radius_display = 0.0
        expansion_ratio = 0.0

        if nozzle_mode == "Set Expansion Ratio":
            expansion_ratio_input = st.number_input(
                "Expansion Ratio (A_exit/A_orifice)",
                min_value=1.0,
                max_value=100.0,
                value=st.session_state['expansion_ratio'],
                step=0.1,
                format="%.2f",
                help="Ratio of the nozzle exit area to the orifice area."
            )
            
            # Calculate exit radius based on the input expansion ratio
            orifice_area = np.pi * (orifice_radius * 1e-3)**2
            exit_area = orifice_area * expansion_ratio_input
            exit_radius_calc = np.sqrt(exit_area / np.pi) * 1e3
            
            # Display calculated exit radius in an info box
            st.markdown(f'''
            <div class="nozzle-info">
                <strong>📏 Calculated Exit Radius:</strong> {exit_radius_calc:.3f} mm<br>
                <strong>📐 Exit Area:</strong> {exit_area*1e6:.3f} mm²<br>
                <strong>🎯 Orifice Area:</strong> {orifice_area*1e6:.3f} mm²
            </div>
            ''', unsafe_allow_html=True)
            
            st.session_state['exit_radius'] = exit_radius_calc
            st.session_state['expansion_ratio'] = expansion_ratio_input
            
            # Assign the local variables to the main scope variables
            expansion_ratio = expansion_ratio_input
            exit_radius_display = exit_radius_calc
            
        else:  # mode is "Set Exit Radius"
            exit_radius_input = st.number_input(
                "Exit Radius (mm)",
                min_value=orifice_radius,  # Must be at least as large as orifice
                max_value=10.0,
                value=max(st.session_state['exit_radius'], orifice_radius),
                step=0.01,
                format="%.3f",
                help="Radius of the nozzle exit."
            )
            
            # Calculate expansion ratio based on orifice and exit radii
            if orifice_radius > 0:
                expansion_ratio_calc = (exit_radius_input / orifice_radius)**2
            else:
                expansion_ratio_calc = 1.0
            
            # Calculate areas
            orifice_area = np.pi * (orifice_radius * 1e-3)**2
            exit_area = np.pi * (exit_radius_input * 1e-3)**2
            
            # Display calculated expansion ratio in an info box
            st.markdown(f'''
            <div class="nozzle-info">
                <strong>📈 Calculated Expansion Ratio:</strong> {expansion_ratio_calc:.2f}<br>
                <strong>📐 Exit Area:</strong> {exit_area*1e6:.3f} mm²<br>
                <strong>🎯 Orifice Area:</strong> {orifice_area*1e6:.3f} mm²
            </div>
            ''', unsafe_allow_html=True)
            
            st.session_state['exit_radius'] = exit_radius_input
            st.session_state['expansion_ratio'] = expansion_ratio_calc
            
            # Assign the local variables to the main scope variables
            expansion_ratio = expansion_ratio_calc
            exit_radius_display = exit_radius_input

        nozzle_efficiency = st.slider(
            "Nozzle Efficiency (%)",
            min_value=50,
            max_value=100,
            value=95,
            step=1,
            help="The percentage of ideal exhaust velocity achieved in practice."
        )

        Q_input = st.number_input(
            "Heat Input Rate (W)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            format="%.1f",
            help="Constant heat added to the chamber (e.g., from a heating element)."
        )

        T_initial = st.number_input(
            "Initial Chamber Temperature (K)",
            min_value=200.0,
            max_value=2000.0,
            value=300.0,
            step=1.0,
            help="The starting temperature of the butane gas in the chamber."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Environment & Simulation Settings ---
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.subheader("Environment & Simulation Settings")

        P_ambient = st.number_input(
            "Ambient Pressure (Pa)",
            min_value=0.0,
            max_value=101325.0,
            value=0.0,
            step=1000.0,
            help="The pressure of the surrounding environment (0 for vacuum)."
        )
        
        total_time = st.number_input(
            "Total Simulation Time (s)",
            min_value=1.0,
            max_value=600.0,
            value=100.0,
            step=10.0,
            help="The total duration of the simulation."
        )
        
        dt = st.number_input(
            "Time Step (s)",
            min_value=0.001,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.3f",
            help="The time increment for each simulation step. A smaller value increases accuracy but slows down the simulation."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        if st.button("Run Simulation", use_container_width=True):
            st.session_state['run_sim'] = True
        else:
            st.session_state['run_sim'] = False
            
        # Display the Cp table in a separate sidebar section
        st.markdown('---')
        st.subheader("Butane $C_p$ Data Table")
        st.info("Specific heat capacity ($C_p$) for Butane as a function of temperature. This data is used to model the thermal response of the thruster.")
        
        # Create a DataFrame for the Cp data
        cp_df = pd.DataFrame({
            "Temperature (K)": T_CP_DATA[:11],  # Show first 11 points
            "Cp (J/kg·K)": CP_DATA[:11]
        })
        st.dataframe(cp_df, height=300)
    
    # Main content area
    if st.session_state.get('run_sim', False):
        st.info("Simulation running...")
        
        # Package parameters for validation and simulation
        params = {
            'T_initial': T_initial,
            'Q_input': Q_input,
            'expansion_ratio': expansion_ratio,
            'P_ambient': P_ambient,
            'total_time': total_time,
            'dt': dt,
            'initial_butane_mass': initial_butane_mass,
            'tank_volume': tank_volume,
            'tank_temperature': tank_temperature,
            'orifice_radius': orifice_radius,
            'discharge_coefficient': discharge_coefficient,
            'nozzle_efficiency': nozzle_efficiency,
        }
        
        validation_messages = validate_simulation_parameters(params)
    
        for message_type, message_text in validation_messages:
            if message_type == "excellent":
                st.markdown(f'<div class="validation-excellent">{message_text}</div>', unsafe_allow_html=True)
            elif message_type == "good":
                st.markdown(f'<div class="validation-good">{message_text}</div>', unsafe_allow_html=True)
            elif message_type == "warning":
                st.markdown(f'<div class="validation-warning">{message_text}</div>', unsafe_allow_html=True)
            elif message_type == "danger":
                st.markdown(f'<div class="validation-danger">{message_text}</div>', unsafe_allow_html=True)
        
        # Run simulation
        results = run_enhanced_simulation(params)
        
        # --- Display Nozzle Geometry Summary ---
        st.markdown('<div class="thermodynamic-box">', unsafe_allow_html=True)
        st.header("🔧 Nozzle Geometry Summary")
        
        geom_col1, geom_col2, geom_col3, geom_col4 = st.columns(4)
        with geom_col1:
            st.metric(label="Orifice Radius", value=f"{orifice_radius:.3f} mm")
        with geom_col2:
            st.metric(label="Exit Radius", value=f"{results['nozzle_geometry']['exit_radius_mm']:.3f} mm")
        with geom_col3:
            st.metric(label="Expansion Ratio", value=f"{expansion_ratio:.2f}")
        with geom_col4:
            st.metric(label="Orifice Area", value=f"{results['nozzle_geometry']['orifice_area']*1e6:.3f} mm²")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Display Results ---
        st.markdown('<div class="performance-summary">', unsafe_allow_html=True)
        st.header("✨ Final Simulation Results")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(label="Final Thrust", value=f"{results['final_values']['thrust']:.3f} N")
        with col2:
            st.metric(label="Final Specific Impulse", value=f"{results['final_values']['specific_impulse']:.1f} s")
        with col3:
            st.metric(label="Final Exhaust Velocity", value=f"{results['final_values']['exhaust_velocity']:.1f} m/s")
        with col4:
            st.metric(label="Final Mass Flow Rate", value=f"{results['final_values']['mass_flow_rate']*1000:.3f} g/s")
        with col5:
            st.metric(label="Final Exit Pressure", value=f"{results['final_values']['exit_pressure']:.0f} Pa")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Plot the Cp table data ---
        st.header("🔬 Butane Specific Heat Capacity ($C_p$)")
        st.info("This plot visualizes the specific heat data used in the simulation to model the thermodynamic behavior of butane.")
        
        fig_cp = go.Figure()
        fig_cp.add_trace(go.Scatter(
            x=T_CP_DATA,
            y=CP_DATA,
            mode='lines+markers',
            name='Specific Heat Capacity',
            line=dict(color='#8A2BE2', width=2),
            marker=dict(size=8, color='#8A2BE2', symbol='circle')
        ))
        
        fig_cp.update_layout(
            title='Specific Heat Capacity (C_p) vs. Temperature',
            xaxis_title='Temperature (K)',
            yaxis_title='Specific Heat C_p (J/kg·K)',
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_cp, use_container_width=True)
        
        # Plotly Charts
        st.header("📈 Simulation Plots Over Time")
        
        fig = make_subplots(rows=3, cols=2,
                            subplot_titles=("Thrust Force (N)", "Specific Impulse (s)",
                                            "Chamber & Exit Pressure (Pa)", "Mass Flow Rate (kg/s)", 
                                            "Chamber Temperature (K)", "Exhaust Velocity (m/s)"))
        
        # Thrust Force
        fig.add_trace(go.Scatter(x=results['times'], y=results['thrust_forces'],
                                    mode='lines', name='Total Thrust', line=dict(color='#f093fb')),
                        row=1, col=1)
        fig.add_trace(go.Scatter(x=results['times'], y=results['momentum_thrusts'],
                                    mode='lines', name='Momentum Thrust', line=dict(dash='dash', color='#667eea')),
                        row=1, col=1)
        fig.add_trace(go.Scatter(x=results['times'], y=results['pressure_thrusts'],
                                    mode='lines', name='Pressure Thrust', line=dict(dash='dot', color='#764ba2')),
                        row=1, col=1)
        
        # Specific Impulse
        fig.add_trace(go.Scatter(x=results['times'], y=results['specific_impulses'],
                                    mode='lines', name='Specific Impulse', line=dict(color='#4facfe')),
                        row=1, col=2)
        
        # Pressures
        fig.add_trace(go.Scatter(x=results['times'], y=results['chamber_pressures'],
                                    mode='lines', name='Chamber Pressure', line=dict(color='#38f9d7')),
                        row=2, col=1)
        fig.add_trace(go.Scatter(x=results['times'], y=results['exit_pressures'],
                                    mode='lines', name='Exit Pressure', line=dict(dash='dash', color='#43e97b')),
                        row=2, col=1)
        fig.add_trace(go.Scatter(x=results['times'], y=results['tank_pressures'],
                                    mode='lines', name='Tank Pressure', line=dict(dash='dot', color='#ff7e5f')),
                        row=2, col=1)
        
        # Mass Flow Rate
        fig.add_trace(go.Scatter(x=results['times'], y=results['mass_flow_rates'],
                                    mode='lines', name='Mass Flow Rate', line=dict(color='#ff7e5f')),
                        row=2, col=2)
        
        # Temperature
        fig.add_trace(go.Scatter(x=results['times'], y=results['temperatures'],
                                    mode='lines', name='Chamber Temperature', line=dict(color='#ffa500')),
                        row=3, col=1)
        
        # Exhaust Velocity
        fig.add_trace(go.Scatter(x=results['times'], y=results['exhaust_velocities'],
                                    mode='lines', name='Exhaust Velocity', line=dict(color='#ff006e')),
                        row=3, col=2)
        
        fig.update_layout(height=1000, title_text="Thruster Performance Over Time", showlegend=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional detailed plots
        st.header("🔍 Detailed Analysis Plots")
        
        fig2 = make_subplots(rows=2, cols=2,
                            subplot_titles=("Pressure Evolution", "Mass Depletion", 
                                          "Specific Heat Variation", "Thrust Components"))
        
        # Pressure comparison
        fig2.add_trace(go.Scatter(x=results['times'], y=results['tank_pressures']/1000,
                                    mode='lines', name='Tank Pressure (kPa)', line=dict(color='#43e97b')),
                        row=1, col=1)
        fig2.add_trace(go.Scatter(x=results['times'], y=results['chamber_pressures']/1000,
                                    mode='lines', name='Chamber Pressure (kPa)', line=dict(color='#38f9d7')),
                        row=1, col=1)
        fig2.add_trace(go.Scatter(x=results['times'], y=results['exit_pressures']/1000,
                                    mode='lines', name='Exit Pressure (kPa)', line=dict(color='#f093fb')),
                        row=1, col=1)
        
        # Butane mass depletion
        fig2.add_trace(go.Scatter(x=results['times'], y=results['butane_masses']*1000,
                                    mode='lines', name='Butane Mass (g)', line=dict(color='#ff7e5f')),
                        row=1, col=2)
        
        # Cp variation
        fig2.add_trace(go.Scatter(x=results['times'], y=results['cp_values'],
                                    mode='lines', name='Cp (J/kg·K)', line=dict(color='#8A2BE2')),
                        row=2, col=1)
        
        # Thrust components
        fig2.add_trace(go.Scatter(x=results['times'], y=results['momentum_thrusts']*1000,
                                    mode='lines', name='Momentum Thrust (mN)', line=dict(color='#667eea')),
                        row=2, col=2)
        fig2.add_trace(go.Scatter(x=results['times'], y=results['pressure_thrusts']*1000,
                                    mode='lines', name='Pressure Thrust (mN)', line=dict(color='#764ba2')),
                        row=2, col=2)
        
        fig2.update_layout(height=800, title_text="Detailed Performance Analysis", showlegend=True, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Dataframe
        st.header("📊 Simulation Data Table")
        df_results = pd.DataFrame({
            "Time (s)": results['times'],
            "Chamber Temp (K)": results['temperatures'],
            "Chamber Press (Pa)": results['chamber_pressures'],
            "Exit Press (Pa)": results['exit_pressures'],
            "Tank Press (Pa)": results['tank_pressures'],
            "Mass Flow Rate (kg/s)": results['mass_flow_rates'],
            "Exhaust Vel (m/s)": results['exhaust_velocities'],
            "Specific Impulse (s)": results['specific_impulses'],
            "Thrust (N)": results['thrust_forces'],
            "Butane Mass (kg)": results['butane_masses']
        })
        st.dataframe(df_results)
        
        # Performance summary
        st.markdown('<div class="thermodynamic-box">', unsafe_allow_html=True)
        st.header("📋 Performance Summary")
        
        max_thrust = np.max(results['thrust_forces'])
        avg_thrust = np.mean(results['thrust_forces'][results['thrust_forces'] > 0])
        max_isp = np.max(results['specific_impulses'])
        avg_isp = np.mean(results['specific_impulses'][results['specific_impulses'] > 0])
        total_impulse = np.trapz(results['thrust_forces'], results['times'])
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Maximum Thrust", f"{max_thrust:.3f} N")
            st.metric("Average Thrust", f"{avg_thrust:.3f} N")
            
        with summary_col2:
            st.metric("Maximum Isp", f"{max_isp:.1f} s")
            st.metric("Average Isp", f"{avg_isp:.1f} s")
            
        with summary_col3:
            st.metric("Total Impulse", f"{total_impulse:.2f} N·s")
            st.metric("Propellant Used", f"{(initial_butane_mass - results['final_values']['tank_pressure']/(R_BUTANE*tank_temperature)*tank_volume*1e-3)*1000:.1f} g")
        
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Configure simulation parameters in the sidebar and click 'Run Simulation' to start.")
        
        # Display some helpful information while waiting
        st.markdown("""
        ### 🎯 Key Features of This Simulation:
        
        - **Unified Orifice Design**: The orifice serves as the flow choking point, equivalent to the nozzle throat
        - **Expansion Ratio Control**: Primary method for nozzle design, with automatic exit radius calculation
        - **Dynamic Exit Pressure**: Calculated based on chamber pressure and expansion ratio using isentropic flow equations
        - **Real Butane Properties**: Temperature-dependent specific heat capacity (Cp) data from NIST
        - **Complete Mass Balance**: Tracks butane consumption and tank pressure drop over time
        - **Thermodynamic Accuracy**: Follows project specifications for all major calculations
        
        ### 🔧 Design Process:
        1. Set your orifice radius (this controls mass flow rate)
        2. Choose expansion ratio (this determines exit pressure and performance)
        3. The app calculates exit radius automatically
        4. Or vice versa - set exit radius and see the required expansion ratio
        """)

# Entry point
if __name__ == "__main__":
    main()