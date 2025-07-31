import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
</style>
""", unsafe_allow_html=True)

# Physical Constants for Butane
GAMMA = 1.05  # Specific heat ratio for butane
R_BUTANE = 143.05  # Gas constant for butane (J/kg·K)
NOZZLE_EFFICIENCY = 0.93  # 93% nozzle efficiency
G0 = 9.80665  # Standard gravity (m/s²)
MOLECULAR_WEIGHT_BUTANE = 58.12  # g/mol

# Fixed Nozzle Geometry (converted to meters)
THROAT_RADIUS_MM = 0.244  # mm
EXIT_RADIUS_MM = 5.62  # mm
A_THROAT_MM2 = np.pi * THROAT_RADIUS_MM**2  # mm²
A_EXIT_MM2 = np.pi * EXIT_RADIUS_MM**2  # mm²
EXPANSION_RATIO = A_EXIT_MM2 / A_THROAT_MM2

# Convert areas to m²
A_THROAT = A_THROAT_MM2 * 1e-6  # m²
A_EXIT = A_EXIT_MM2 * 1e-6  # m²

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

def compute_exhaust_velocity_isentropic(T_chamber, P_chamber, P_exit, gamma=GAMMA, R=R_BUTANE, efficiency=NOZZLE_EFFICIENCY):
    """
    Compute exhaust velocity using isentropic flow equations.
    
    V₂ = η * √(2γR*T₁/(γ-1) * [1 - (P₂/P₁)^((γ-1)/γ)])
    
    Args:
        T_chamber: Chamber temperature (K)
        P_chamber: Chamber pressure (Pa)
        P_exit: Exit pressure (Pa)
        gamma: Specific heat ratio
        R: Gas constant (J/kg·K)
        efficiency: Nozzle efficiency factor (0-1)
    
    Returns:
        Exhaust velocity (m/s)
    """
    if P_chamber <= P_exit:
        return 0.0  # No flow possible
    
    pressure_ratio = P_exit / P_chamber
    
    # Prevent numerical issues with very small pressure ratios
    pressure_ratio = max(pressure_ratio, 1e-10)
    
    # Isentropic expansion formula
    term1 = (2 * gamma * R * T_chamber) / (gamma - 1)
    term2 = 1 - pressure_ratio**((gamma - 1) / gamma)
    
    if term2 < 0:
        term2 = 0  # Prevent negative values under square root
    
    v_ideal = np.sqrt(term1 * term2)
    
    # Apply nozzle efficiency
    v_actual = v_ideal * efficiency
    
    return v_actual

def compute_specific_impulse(exhaust_velocity):
    """
    Compute specific impulse from exhaust velocity.
    Isp = V₂ / g₀
    
    Args:
        exhaust_velocity: Exhaust velocity (m/s)
    
    Returns:
        Specific impulse (s)
    """
    return exhaust_velocity / G0

def compute_thrust_force(mass_flow_rate, exhaust_velocity, P_exit, P_ambient, A_exit):
    """
    Compute thrust force using momentum and pressure thrust.
    F = ṁ*V₂ + (P₂ - P₃)*A₂
    
    Args:
        mass_flow_rate: Mass flow rate (kg/s)
        exhaust_velocity: Exhaust velocity (m/s)
        P_exit: Exit pressure (Pa)
        P_ambient: Ambient pressure (Pa)
        A_exit: Exit area (m²)
    
    Returns:
        Thrust force (N), momentum thrust (N), pressure thrust (N)
    """
    momentum_thrust = mass_flow_rate * exhaust_velocity
    pressure_thrust = (P_exit - P_ambient) * A_exit
    total_thrust = momentum_thrust + pressure_thrust
    
    return total_thrust, momentum_thrust, pressure_thrust

def update_chamber_temperature(T_current, Q_input, mass_flow_rate, dt, cp_func):
    """
    Update chamber temperature based on energy balance.
    Q = ṁ * Cp * dT/dt
    dT/dt = Q / (ṁ * Cp)
    
    Args:
        T_current: Current temperature (K)
        Q_input: Heat input rate (W)
        mass_flow_rate: Mass flow rate (kg/s)
        dt: Time step (s)
        cp_func: Function to get Cp at given temperature
    
    Returns:
        New temperature (K)
    """
    if mass_flow_rate <= 0:
        return T_current
    
    cp_current = cp_func(T_current)
    dT_dt = Q_input / (mass_flow_rate * cp_current)
    T_new = T_current + dT_dt * dt
    
    # Ensure temperature doesn't go below absolute zero or become unrealistic
    T_new = max(T_new, 200)  # Minimum reasonable temperature
    T_new = min(T_new, 3000)  # Maximum reasonable temperature
    
    return T_new

def compute_chamber_pressure_choked_flow(T_chamber, mass_flow_rate, A_throat, gamma=GAMMA, R=R_BUTANE):
    """
    Compute chamber pressure for choked flow conditions.
    For choked flow: ṁ = P₁ * A* * √(γ/RT₁) * (2/(γ+1))^((γ+1)/(2(γ-1)))
    
    Args:
        T_chamber: Chamber temperature (K)
        mass_flow_rate: Mass flow rate (kg/s)
        A_throat: Throat area (m²)
        gamma: Specific heat ratio
        R: Gas constant (J/kg·K)
    
    Returns:
        Chamber pressure (Pa)
    """
    if T_chamber <= 0 or mass_flow_rate <= 0 or A_throat <= 0:
        return 0.0
    
    # Choked flow coefficient
    term1 = np.sqrt(gamma / (R * T_chamber))
    term2 = (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))
    choked_coeff = term1 * term2
    
    # Solve for chamber pressure
    P_chamber = mass_flow_rate / (A_throat * choked_coeff)
    
    return P_chamber

def compute_exit_mach_number(P_chamber, P_exit, gamma=GAMMA):
    """
    Compute exit Mach number from pressure ratio.
    
    Args:
        P_chamber: Chamber pressure (Pa)
        P_exit: Exit pressure (Pa)
        gamma: Specific heat ratio
    
    Returns:
        Exit Mach number
    """
    if P_chamber <= P_exit:
        return 0.0
    
    pressure_ratio = P_exit / P_chamber
    
    # For isentropic flow: P₂/P₁ = (1 + (γ-1)/2 * M²)^(-γ/(γ-1))
    # Solving for M: M = √(2/(γ-1) * [(P₁/P₂)^((γ-1)/γ) - 1])
    term = (1 / pressure_ratio)**((gamma - 1) / gamma) - 1
    if term <= 0:
        return 0.0
    
    mach = np.sqrt((2 / (gamma - 1)) * term)
    return mach

def validate_simulation_parameters(params):
    """
    Validate simulation parameters and return warnings/errors.
    
    Args:
        params: Dictionary of simulation parameters
    
    Returns:
        List of validation messages
    """
    messages = []
    
    # Temperature validation
    if params['T_initial'] < 200:
        messages.append(("danger", "Initial temperature too low (< 200K). May cause unrealistic results."))
    elif params['T_initial'] > 1500:
        messages.append(("warning", "Initial temperature very high (> 1500K). Ensure realistic for your application."))
    elif params['T_initial'] > 500:
        messages.append(("good", "Initial temperature in good range for resistojet operation."))
    else:
        messages.append(("excellent", "Initial temperature in excellent range for cold gas thrusters."))
    
    # Mass flow rate validation
    if params['mass_flow_rate'] < 0.001:
        messages.append(("warning", "Very low mass flow rate. May not sustain stable combustion."))
    elif params['mass_flow_rate'] > 0.050:
        messages.append(("warning", "High mass flow rate. Ensure adequate heat input."))
    else:
        messages.append(("good", "Mass flow rate in reasonable range."))
    
    # Pressure ratio validation
    P_ratio = params['P_ambient'] / params['P_exit']
    if P_ratio > 0.5:
        messages.append(("danger", "Ambient pressure too close to exit pressure. May cause flow separation."))
    elif P_ratio > 0.1:
        messages.append(("warning", "High ambient to exit pressure ratio. Check nozzle design."))
    else:
        messages.append(("good", "Good pressure ratio for efficient expansion."))
    
    # Heat input validation
    heat_per_mass = params['Q_input'] / (params['mass_flow_rate'] / 1000)  # W per kg/s
    if heat_per_mass < 1000:
        messages.append(("warning", "Low specific heat input. Temperature rise may be minimal."))
    elif heat_per_mass > 50000:
        messages.append(("danger", "Very high specific heat input. May cause unrealistic temperatures."))
    else:
        messages.append(("good", "Heat input appropriate for mass flow rate."))
    
    return messages 

def compute_theoretical_performance(T_chamber, P_chamber, P_exit):
    """
    Compute theoretical performance metrics.
    
    Args:
        T_chamber: Chamber temperature (K)
        P_chamber: Chamber pressure (Pa)
        P_exit: Exit pressure (Pa)
        mass_flow_rate: Mass flow rate (kg/s)
    
    Returns:
        Dictionary of theoretical performance metrics
    """
    # Exit Mach number
    M_exit = compute_exit_mach_number(P_chamber, P_exit, GAMMA)
    
    # Exit temperature (isentropic relation)
    T_exit = T_chamber * (P_exit / P_chamber)**((GAMMA - 1) / GAMMA)
    
    # Characteristic velocity (c*)
    c_star = np.sqrt(GAMMA * R_BUTANE * T_chamber) / GAMMA * np.sqrt((2 / (GAMMA + 1))**((GAMMA + 1) / (GAMMA - 1)))
    
    # Thrust coefficient
    pressure_ratio = P_exit / P_chamber
    Cf = np.sqrt(2 * GAMMA**2 / (GAMMA - 1) * (2 / (GAMMA + 1))**((GAMMA + 1) / (GAMMA - 1)) * 
                (1 - pressure_ratio**((GAMMA - 1) / GAMMA)))
    
    return {
        'exit_mach': M_exit,
        'exit_temperature': T_exit,
        'characteristic_velocity': c_star,
        'thrust_coefficient': Cf
    }

def run_enhanced_simulation(params):
    """
    Run the enhanced thruster simulation with realistic thermodynamics.
    
    Args:
        params: Dictionary of simulation parameters
    
    Returns:
        Dictionary containing comprehensive results
    """
    # Extract parameters
    T_initial = params['T_initial']
    Q_input = params['Q_input']
    mass_flow_rate = params['mass_flow_rate'] / 1000  # Convert g/s to kg/s
    P_exit = params['P_exit'] * 1e6  # Convert MPa to Pa
    P_ambient = params['P_ambient'] * 1e6  # Convert MPa to Pa
    total_time = params['total_time']
    dt = params['dt']
    
    # Initialize arrays
    times = np.arange(0, total_time + dt, dt)
    n_steps = len(times)
    
    # Result arrays
    temperatures = np.zeros(n_steps)
    chamber_pressures = np.zeros(n_steps)
    exhaust_velocities = np.zeros(n_steps)
    specific_impulses = np.zeros(n_steps)
    thrust_forces = np.zeros(n_steps)
    momentum_thrusts = np.zeros(n_steps)
    pressure_thrusts = np.zeros(n_steps)
    cp_values = np.zeros(n_steps)
    mach_numbers = np.zeros(n_steps)
    exit_temperatures = np.zeros(n_steps)
    
    # Initial conditions
    T_chamber = T_initial
    
    # Simulation loop
    for i in range(n_steps):
        # Store current temperature
        temperatures[i] = T_chamber
        
        # Get current Cp
        cp_current = get_cp(T_chamber)
        cp_values[i] = cp_current
        
        # Compute chamber pressure for choked flow
        P_chamber = compute_chamber_pressure_choked_flow(T_chamber, mass_flow_rate, A_THROAT, GAMMA, R_BUTANE)
        chamber_pressures[i] = P_chamber
        
        # Compute exhaust velocity
        v_exhaust = compute_exhaust_velocity_isentropic(T_chamber, P_chamber, P_exit, GAMMA, R_BUTANE, NOZZLE_EFFICIENCY)
        exhaust_velocities[i] = v_exhaust
        
        # Compute specific impulse
        isp = compute_specific_impulse(v_exhaust)
        specific_impulses[i] = isp
        
        # Compute thrust components
        F_total, F_momentum, F_pressure = compute_thrust_force(mass_flow_rate, v_exhaust, P_exit, P_ambient, A_EXIT)
        thrust_forces[i] = F_total
        momentum_thrusts[i] = F_momentum
        pressure_thrusts[i] = F_pressure
        
        # Compute additional metrics
        M_exit = compute_exit_mach_number(P_chamber, P_exit, GAMMA)
        mach_numbers[i] = M_exit
        
        T_exit = T_chamber * (P_exit / P_chamber)**((GAMMA - 1) / GAMMA) if P_chamber > 0 else T_chamber
        exit_temperatures[i] = T_exit
        
        # Update temperature for next time step (except for last iteration)
        if i < n_steps - 1:
            T_chamber = update_chamber_temperature(T_chamber, Q_input, mass_flow_rate, dt, get_cp)
    
    # Compute theoretical performance for final state
    theoretical = compute_theoretical_performance(temperatures[-1], chamber_pressures[-1], P_exit)
    
    return {
        'times': times,
        'temperatures': temperatures,
        'chamber_pressures': chamber_pressures / 1e6,  # Convert back to MPa for display
        'exhaust_velocities': exhaust_velocities,
        'specific_impulses': specific_impulses,
        'thrust_forces': thrust_forces,
        'momentum_thrusts': momentum_thrusts,
        'pressure_thrusts': pressure_thrusts,
        'cp_values': cp_values,
        'mach_numbers': mach_numbers,
        'exit_temperatures': exit_temperatures,
        'mass_flow_rate': mass_flow_rate,
        'theoretical': theoretical,
        'final_values': {
            'temperature': temperatures[-1],
            'chamber_pressure': chamber_pressures[-1] / 1e6,
            'exhaust_velocity': exhaust_velocities[-1],
            'specific_impulse': specific_impulses[-1],
            'thrust': thrust_forces[-1],
            'cp': cp_values[-1],
            'mach_number': mach_numbers[-1],
            'exit_temperature': exit_temperatures[-1]
        }
    }

# Streamlit Application
def main():
    # Header
    st.markdown("""
    <div class="hero-header">
        <h1>🚀 Butane Thruster Simulation</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Thermodynamic modeling with realistic Isp calculations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input parameters
    with st.sidebar:
        st.header("🔧 Simulation Parameters")
        
        # Thermodynamic parameters
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.subheader("Thermodynamic Properties")
        
        T_initial = st.number_input(
            "Initial Chamber Temperature (K)",
            min_value=200.0,
            max_value=2000.0,
            value=350.0,
            step=0.1,
            format="%.2f", 
            help="Starting temperature of the combustion chamber"
        )
        
        Q_input = st.number_input(
            "Heat Input Rate (W)",
            min_value=1.0,
            max_value=200.0,
            value=20.0,
            step=0.1,
            format="%.2f", 
            help="Constant heat input to the thruster"
        )
        
        mass_flow_rate = st.number_input(
            "Mass Flow Rate (g/s)",
            min_value=0.001,
            max_value=0.100,
            value=0.008,
            step=0.001,
            format="%.3f",
            help="Propellant mass flow rate"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Pressure parameters
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.subheader("Pressure Conditions")
        
        P_exit = st.number_input(
            "Nozzle Exit Pressure (MPa)",
            min_value=0.001,
            max_value=1.0,
            value=0.002,
            step=0.001,
            format="%.3f",
            help="Pressure at nozzle exit (P₂)"
        )
        
        P_ambient = st.number_input(
            "Ambient Pressure (MPa)",
            min_value=0.000001,
            max_value=0.1013,
            value=0.0001,
            step=0.000001,
            format="%.6f",
            help="Environmental pressure (P₃)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Simulation parameters
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.subheader("Simulation Settings")
        
        total_time = st.number_input(
            "Simulation Duration (s)",
            min_value=1.0,
            max_value=600.0,
            value=30.0,
            step=0.1,
            format="%.2f", 

            help="Total simulation time"
        )
        
        dt = st.selectbox(
            "Time Step (s)",
            options=[0.01, 0.05, 0.1, 0.5, 1.0],
            index=2,
            help="Simulation time step resolution"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fixed parameters display
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.subheader("Fixed Parameters")
        st.markdown(f"**Throat Radius:** {THROAT_RADIUS_MM:.3f} mm")
        st.markdown(f"**Exit Radius:** {EXIT_RADIUS_MM:.3f} mm")
        st.markdown(f"**Expansion Ratio:** {EXPANSION_RATIO:.1f}")
        st.markdown(f"**Nozzle Efficiency:** {NOZZLE_EFFICIENCY*100:.0f}%")
        st.markdown(f"**k (Butane):** {GAMMA}")
        st.markdown(f"**R (Butane):** {R_BUTANE:.1f} J/kg·K")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        run_simulation = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
        
        if st.button("📊 Show Cp Lookup Table", use_container_width=True):
            st.subheader("Butane Cp Lookup Table")
            cp_df = pd.DataFrame({
                'Temperature (K)': T_CP_DATA,
                'Cp (J/kg·K)': CP_DATA
            })
            st.dataframe(cp_df, height=300)
        
        # Parameter validation
        sim_params = {
            'T_initial': T_initial,
            'Q_input': Q_input,
            'mass_flow_rate': mass_flow_rate,
            'P_exit': P_exit,
            'P_ambient': P_ambient,
            'total_time': total_time,
            'dt': dt
        }
        
        validation_messages = validate_simulation_parameters(sim_params)
        
        st.subheader("⚠️ Parameter Validation")
        for msg_type, message in validation_messages:
            st.markdown(f'<div class="validation-{msg_type}">{message}</div>', unsafe_allow_html=True)
    
    with col1:
        if run_simulation:
            # Run simulation
            with st.spinner("🔄 Running enhanced simulation..."):
                results = run_enhanced_simulation(sim_params)
            
            st.success("✅ Simulation completed successfully!")
            
            # Performance summary
            st.markdown("""
            <div class="performance-summary">
                <h3>🎯 Performance Summary</h3>
                <p>Final thruster performance after thermal equilibrium</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display key results
            col1a, col2a, col3a, col4a, col5a = st.columns(5)
            
            with col1a:
                st.metric(
                    "Thrust", 
                    f"{results['final_values']['thrust']:.4f} N",
                    help="Total thrust force"
                )
            
            with col2a:
                st.metric(
                    "Specific Impulse", 
                    f"{results['final_values']['specific_impulse']:.1f} s",
                    help="Isp = V₂ / g₀"
                )
            
            with col3a:
                st.metric(
                    "Exhaust Velocity", 
                    f"{results['final_values']['exhaust_velocity']:.0f} m/s",
                    help="Nozzle exit velocity"
                )
            
            with col4a:
                st.metric(
                    "Chamber Pressure", 
                    f"{results['final_values']['chamber_pressure']:.3f} MPa",
                    help="Computed from choked flow"
                )
            
            with col5a:
                st.metric(
                    "Exit Mach", 
                    f"{results['final_values']['mach_number']:.2f}",
                    help="Exit Mach number"
                )
            
            # Additional performance metrics
            st.subheader("🔬 Advanced Performance Metrics")
            col1b, col2b, col3b, col4b = st.columns(4)
            
            with col1b:
                st.metric(
                    "Exit Temperature",
                    f"{results['final_values']['exit_temperature']:.0f} K",
                    help="Gas temperature at nozzle exit"
                )
            
            with col2b:
                st.metric(
                    "Characteristic Velocity",
                    f"{results['theoretical']['characteristic_velocity']:.0f} m/s",
                    help="c* - combustion efficiency indicator"
                )
            
            with col3b:
                st.metric(
                    "Thrust Coefficient",
                    f"{results['theoretical']['thrust_coefficient']:.3f}",
                    help="Cf - nozzle performance indicator"
                )
            
            with col4b:
                thrust_to_weight = results['final_values']['thrust'] / (results['mass_flow_rate'] * G0)
                st.metric(
                    "Thrust-to-Weight",
                    f"{thrust_to_weight:.1f}",
                    help="Thrust per unit weight flow"
                )
            
            # Create comprehensive plots
            fig = make_subplots(
                rows=5, cols=2,
                subplot_titles=(
                    "Chamber Temperature vs Time",
                    "Specific Impulse vs Time", 
                    "Total Thrust vs Time",
                    "Component Thrust vs Time",
                    "Chamber Pressure vs Time",
                    "Exhaust Velocity vs Time",
                    "Exit Mach Number vs Time",
                    "Specific Heat Capacity vs Time",
                    "Exit Temperature vs Time",
                    ""  # Empty subplot for 5x2 grid
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": True}],  # Component thrust needs secondary y-axis
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ]
            )
            
            # =============================================================================
            # ROW 1: TEMPERATURE AND SPECIFIC IMPULSE
            # =============================================================================
            fig.add_trace(
                go.Scatter(
                    x=results['times'], 
                    y=results['temperatures'], 
                    name="Chamber Temperature",
                    line=dict(color='#ff6b6b', width=3)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=results['times'], 
                    y=results['specific_impulses'], 
                    name="Specific Impulse",
                    line=dict(color='#4ecdc4', width=3)
                ),
                row=1, col=2
            )
            
            # =============================================================================
            # ROW 2: TOTAL THRUST AND COMPONENT THRUST
            # =============================================================================
            fig.add_trace(
                go.Scatter(
                    x=results['times'], 
                    y=results['thrust_forces'], 
                    name="Total Thrust",
                    line=dict(color='#45b7d1', width=3)
                ),
                row=2, col=1
            )
            
            # Component thrust traces (using secondary y-axis for better visualization)
            fig.add_trace(
                go.Scatter(
                    x=results['times'], 
                    y=results['momentum_thrusts'], 
                    name="Momentum Thrust",
                    line=dict(color='#96ceb4', width=2, dash='dash')
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=results['times'], 
                    y=results['pressure_thrusts'], 
                    name="Pressure Thrust",
                    line=dict(color='#feca57', width=2, dash='dot')
                ),
                row=2, col=2, secondary_y=True
            )
            
            # =============================================================================
            # ROW 3: CHAMBER PRESSURE AND EXHAUST VELOCITY
            # =============================================================================
            fig.add_trace(
                go.Scatter(
                    x=results['times'], 
                    y=results['chamber_pressures'], 
                    name="Chamber Pressure",
                    line=dict(color='#ff9ff3', width=3)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=results['times'], 
                    y=results['exhaust_velocities'], 
                    name="Exhaust Velocity",
                    line=dict(color='#54a0ff', width=3)
                ),
                row=3, col=2
            )
            
            # =============================================================================
            # ROW 4: EXIT MACH NUMBER AND SPECIFIC HEAT CAPACITY
            # =============================================================================
            fig.add_trace(
                go.Scatter(
                    x=results['times'], 
                    y=results['mach_numbers'], 
                    name="Exit Mach Number",
                    line=dict(color='#ff6348', width=3)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=results['times'], 
                    y=results['cp_values'], 
                    name="Specific Heat Capacity",
                    line=dict(color='#5f27cd', width=3)
                ),
                row=4, col=2
            )
            
            # =============================================================================
            # ROW 5: EXIT TEMPERATURE (LEFT SUBPLOT ONLY)
            # =============================================================================
            fig.add_trace(
                go.Scatter(
                    x=results['times'], 
                    y=results['exit_temperatures'], 
                    name="Exit Temperature",
                    line=dict(color='#ff9f43', width=3)
                ),
                row=5, col=1
            )
            
            # =============================================================================
            # LAYOUT CONFIGURATION
            # =============================================================================
            fig.update_layout(
                height=1500,
                showlegend=True,
                title_text="Comprehensive Thruster Performance Analysis",
                title_x=0.5,
                font=dict(size=11),
                legend=dict(
                    x=1.05, 
                    y=1,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                )
            )
            
            # =============================================================================
            # Y-AXIS LABELS CONFIGURATION
            # =============================================================================
            # Row 1: Temperature and Isp
            fig.update_yaxes(title_text="Temperature (K)", row=1, col=1)
            fig.update_yaxes(title_text="Isp (s)", row=1, col=2)
            
            # Row 2: Thrust components
            fig.update_yaxes(title_text="Total Thrust (N)", row=2, col=1)
            fig.update_yaxes(title_text="Momentum Thrust (N)", row=2, col=2)
            fig.update_yaxes(title_text="Pressure Thrust (N)", row=2, col=2, secondary_y=True)
            
            # Row 3: Pressure and velocity
            fig.update_yaxes(title_text="Pressure (MPa)", row=3, col=1)
            fig.update_yaxes(title_text="Velocity (m/s)", row=3, col=2)
            
            # Row 4: Mach and heat capacity
            fig.update_yaxes(title_text="Mach Number", row=4, col=1)
            fig.update_yaxes(title_text="Cp (J/kg·K)", row=4, col=2)
            
            # Row 5: Exit temperature
            fig.update_yaxes(title_text="Temperature (K)", row=5, col=1)
            
            # =============================================================================
            # X-AXIS LABELS CONFIGURATION
            # =============================================================================  
            fig.update_xaxes(title_text="Time (s)")
            
            # =============================================================================
            # DISPLAY THE PLOT
            # =============================================================================
            st.plotly_chart(fig, use_container_width=True)

            # Energy analysis
            st.subheader("⚡ Energy Analysis")
            total_energy_input = Q_input * total_time  # Joules
            kinetic_energy_per_second = 0.5 * results['mass_flow_rate'] * results['final_values']['exhaust_velocity']**2
            efficiency = (kinetic_energy_per_second / Q_input) * 100 if Q_input > 0 else 0
            
            col1c, col2c, col3c = st.columns(3)
            with col1c:
                st.metric("Total Energy Input", f"{total_energy_input:.0f} J", help="Q × time")
            with col2c:
                st.metric("Kinetic Power Output", f"{kinetic_energy_per_second:.2f} W", help="½ṁV²")
            with col3c:
                st.metric("Thermal Efficiency", f"{efficiency:.1f}%", help="Kinetic power / Heat input")
            
            # Detailed results table
            st.subheader("📋 Detailed Simulation Results")
            results_df = pd.DataFrame({
                'Time (s)': results['times'],
                'Temperature (K)': results['temperatures'],
                'Chamber Pressure (MPa)': results['chamber_pressures'],
                'Exhaust Velocity (m/s)': results['exhaust_velocities'],
                'Specific Impulse (s)': results['specific_impulses'],
                'Total Thrust (N)': results['thrust_forces'],
                'Momentum Thrust (N)': results['momentum_thrusts'],
                'Pressure Thrust (N)': results['pressure_thrusts'],
                'Exit Mach': results['mach_numbers'],
                'Exit Temperature (K)': results['exit_temperatures'],
                'Cp (J/kg·K)': results['cp_values']
            })
            
            st.dataframe(
                results_df.style.format({
                    'Time (s)': '{:.2f}',
                    'Temperature (K)': '{:.1f}',
                    'Chamber Pressure (MPa)': '{:.4f}',
                    'Exhaust Velocity (m/s)': '{:.2f}',
                    'Specific Impulse (s)': '{:.2f}',
                    'Total Thrust (N)': '{:.5f}',
                    'Momentum Thrust (N)': '{:.5f}',
                    'Pressure Thrust (N)': '{:.6f}',
                    'Exit Mach': '{:.3f}',
                    'Exit Temperature (K)': '{:.1f}',
                    'Cp (J/kg·K)': '{:.0f}'
                }),
                height=400,
                use_container_width=True
            )
            
            # Comparison with theoretical values
            st.subheader("📊 Theoretical vs Actual Comparison")
            theoretical_isp = results['theoretical']['characteristic_velocity'] * results['theoretical']['thrust_coefficient'] / G0
            actual_isp = results['final_values']['specific_impulse']
            
            comparison_df = pd.DataFrame({
                'Metric': ['Specific Impulse (s)', 'Thrust Coefficient', 'Exit Mach Number'],
                'Theoretical': [theoretical_isp, results['theoretical']['thrust_coefficient'], results['theoretical']['exit_mach']],
                'Actual': [actual_isp, results['theoretical']['thrust_coefficient'], results['final_values']['mach_number']],
                'Difference (%)': [
                    ((actual_isp - theoretical_isp) / theoretical_isp * 100) if theoretical_isp > 0 else 0,
                    0,  # Thrust coefficient is the same
                    ((results['final_values']['mach_number'] - results['theoretical']['exit_mach']) / results['theoretical']['exit_mach'] * 100) if results['theoretical']['exit_mach'] > 0 else 0
                ]
            })
            
            st.dataframe(
                comparison_df.style.format({
                    'Theoretical': '{:.3f}',
                    'Actual': '{:.3f}',
                    'Difference (%)': '{:.1f}'
                }),
                use_container_width=True
            )
            
            # Download results
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_data,
                file_name=f"butane_thruster_simulation_{int(total_time)}s.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Performance recommendations
            st.subheader("💡 Performance Recommendations")
            
            recommendations = []
            
            if results['final_values']['specific_impulse'] < 50:
                recommendations.append("Consider increasing chamber temperature for better Isp")
            elif results['final_values']['specific_impulse'] > 80:
                recommendations.append("Excellent Isp achieved - good thermal management")
            
            if results['final_values']['mach_number'] < 2:
                recommendations.append("Exit Mach number is low - consider optimizing nozzle design")
            elif results['final_values']['mach_number'] > 4:
                recommendations.append("High exit Mach achieved - efficient expansion")
            
            if efficiency < 5:
                recommendations.append("Low thermal efficiency - check heat transfer and insulation")
            elif efficiency > 15:
                recommendations.append("Good thermal efficiency achieved")
            
            pressure_ratio = results['final_values']['chamber_pressure'] / (P_exit)
            if pressure_ratio < 10:
                recommendations.append("Low pressure ratio - consider increasing chamber pressure")
            elif pressure_ratio > 100:
                recommendations.append("High pressure ratio - excellent expansion potential")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
        
        else:
            st.info("🎯 Configure parameters in the sidebar and click 'Run Simulation' to begin analysis.")
            
            # Show sample Cp curve
            T_range = np.linspace(200, 2000, 100)
            cp_range = [get_cp(T) for T in T_range]
            
            fig_cp = go.Figure()
            fig_cp.add_trace(
                go.Scatter(
                    x=T_range, 
                    y=cp_range, 
                    mode='lines',
                    name='Cp(T)',
                    line=dict(color='#e74c3c', width=3)
                )
            )
            fig_cp.add_scatter(
                x=T_CP_DATA, 
                y=CP_DATA, 
                mode='markers',
                name='Data Points',
                marker=dict(color='#3498db', size=8)
            )
            
            fig_cp.update_layout(
                title="Butane Specific Heat Capacity vs Temperature",
                xaxis_title="Temperature (K)",
                yaxis_title="Cp (J/kg·K)",
                height=400
            )
            
            st.plotly_chart(fig_cp, use_container_width=True)
            
            # Show example thermodynamic relationships
            st.subheader("🔬 Thermodynamic Relationships")
            
            st.markdown("""
            **Key Relationships in Butane Thruster Operation:**
            
            1. **Energy Balance**: Q = ṁ × Cp × dT/dt
               - Heat input raises chamber temperature
               - Higher Cp requires more energy for same temperature rise
            
            2. **Choked Flow**: ṁ = P₁ × A* × √(γ/RT₁) × (2/(γ+1))^((γ+1)/(2(γ-1)))
               - Mass flow determines chamber pressure for given throat area
               - Higher temperature reduces required pressure
            
            3. **Isentropic Expansion**: V₂ = √(2γRT₁/(γ-1) × [1 - (P₂/P₁)^((γ-1)/γ)])
               - Higher chamber temperature increases exhaust velocity
               - Lower exit pressure improves expansion efficiency
            
            4. **Thrust Generation**: F = ṁV₂ + (P₂ - P₃)A₂
               - Momentum thrust dominates in vacuum conditions
               - Pressure thrust significant at high ambient pressures
            """)

if __name__ == "__main__":
    main()