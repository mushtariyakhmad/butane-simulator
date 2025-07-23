import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
import io
import base64

# Configure page
st.set_page_config(
    page_title=" Butane Tank Thermal Simulation with Specific Impulse", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .insulated-box {
        background: #e8f4fd;
        border: 2px solid #007bff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .thrust-box {
        background: #fff0e6;
        border: 2px solid #ff6600;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .isp-box {
        background: #f0f8ff;
        border: 2px solid #4169e1;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Enhanced Butane Tank Thermal Simulation with Advanced Isp</h1>
    <p style="text-align: center; color: white; margin: 0;">
        Advanced specific impulse calculations with constant and variable modes
    </p>
</div>
""", unsafe_allow_html=True)

# Physical constants
R = 8.314  # J/(mol·K)
M_butane = 58.1222e-3  # kg/mol
T_critical = 425.13  # K
P_critical = 3.796e6  # Pa
T_boiling = 272.65  # K
T_triple = 134.87  # K
h_vap_normal = 22.389e3  # J/mol at 272.05 K
h_vap_normal_kg = h_vap_normal / M_butane  # J/kg
c_p_liquid = 2.4e3  # J/(kg·K)
gamma = 1.10
g0 = 9.80665  # Standard gravity (m/s²)

# Tank material properties (specific heat capacity in J/kg·K)
TANK_MATERIALS = {
    "Aluminum 6061": {
        "heat_capacity": 896,  # J/(kg·K)
        "density": 2700,  # kg/m³
        "thermal_conductivity": 167  # W/(m·K)
    },
    "Stainless Steel 304": {
        "heat_capacity": 500,  # J/(kg·K)
        "density": 8000,  # kg/m³
        "thermal_conductivity": 16.2  # W/(m·K)
    },
    "Stainless Steel 316": {
        "heat_capacity": 500,  # J/(kg·K)
        "density": 8000,  # kg/m³
        "thermal_conductivity": 16.3  # W/(m·K)
    },
    "Stainless Steel 316L": {
        "heat_capacity": 500,  # J/(kg·K)
        "density": 8000,  # kg/m³
        "thermal_conductivity": 16.3  # W/(m·K)
    },
    "TC-4 Titanium": {
        "heat_capacity": 523,  # J/(kg·K)
        "density": 4430,  # kg/m³
        "thermal_conductivity": 6.7  # W/(m·K)
    }
}

# Antoine equation parameters
A_antoine = 4.70812
B_antoine = 1200.475
C_antoine = -13.013

@st.cache_data
def antoine_vapor_pressure(T):
    """Calculate vapor pressure using Antoine equation"""
    if T < 135 or T > 430:
        return 0
    log_p_bar = A_antoine - B_antoine / (T + C_antoine)
    return 10 ** log_p_bar * 1e5  # Pa

@st.cache_data
def calculate_enhanced_isp(p_chamber, T_chamber, p_ambient, gamma, R_specific, 
                          nozzle_efficiency=1.0, c_star_efficiency=0.95):
    """
    Enhanced specific impulse calculation with efficiency factors
    
    Parameters:
    - p_chamber: Chamber pressure (Pa)
    - T_chamber: Chamber temperature (K)
    - p_ambient: Ambient pressure (Pa)
    - gamma: Specific heat ratio
    - R_specific: Specific gas constant (J/kg·K)
    - nozzle_efficiency: Nozzle efficiency (0-1)
    - c_star_efficiency: Combustion efficiency (0-1)
    """
    
    if p_chamber <= p_ambient or p_chamber <= 0:
        return {
            'theoretical_isp': 0,
            'actual_isp': 0,
            'c_star': 0,
            'c_f': 0,
            'exit_velocity': 0,
            'exit_pressure': p_ambient,
            'exit_temperature': T_chamber,
            'is_choked': False,
            'pressure_ratio': 1.0,
            'expansion_ratio': 1.0
        }
    
    # Pressure ratio
    pressure_ratio = p_ambient / p_chamber
    
    # Critical pressure ratio for choked flow
    gamma_term = gamma / (gamma - 1)
    p_crit_ratio = (2 / (gamma + 1)) ** gamma_term
    
    # Check if flow is choked
    is_choked = pressure_ratio <= p_crit_ratio
    
    # Calculate characteristic velocity (c*)
    c_star_ideal = math.sqrt(gamma * R_specific * T_chamber) / gamma * \
                   ((gamma + 1) / 2) ** ((gamma + 1) / (2 * (gamma - 1)))
    c_star_actual = c_star_ideal * c_star_efficiency
    
    if is_choked:
        # Choked flow conditions
        p_exit = p_chamber * p_crit_ratio
        T_exit = T_chamber * (2 / (gamma + 1))
        
        # Ideal exit velocity for choked flow
        v_exit_ideal = math.sqrt(gamma * R_specific * T_exit)
        
        # Thrust coefficient for choked flow
        c_f_ideal = math.sqrt(gamma * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))
        
    else:
        # Non-choked (overexpanded or underexpanded) flow
        p_exit = p_ambient
        T_exit = T_chamber * (pressure_ratio ** ((gamma - 1) / gamma))
        
        # Ideal exit velocity for non-choked flow
        v_exit_ideal = math.sqrt(2 * gamma * R_specific * T_chamber / (gamma - 1) * 
                                (1 - pressure_ratio ** ((gamma - 1) / gamma)))
        
        # Thrust coefficient for non-choked flow
        c_f_ideal = math.sqrt(2 * gamma**2 / (gamma - 1) * 
                             (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)) * 
                             (1 - pressure_ratio ** ((gamma - 1) / gamma)))
    
    # Apply nozzle efficiency to exit velocity
    v_exit_actual = v_exit_ideal * math.sqrt(nozzle_efficiency)
    c_f_actual = c_f_ideal * math.sqrt(nozzle_efficiency)
    
    # Calculate specific impulse
    isp_theoretical = v_exit_ideal / g0
    isp_actual = v_exit_actual / g0
    
    # Calculate expansion ratio (Area_exit / Area_throat)
    if is_choked and p_exit > 0:
        # For choked flow, calculate ideal expansion ratio
        expansion_ratio = (1 / pressure_ratio) * \
                         ((2 + (gamma - 1) * (p_exit / p_chamber)) / (gamma + 1)) ** \
                         ((gamma + 1) / (2 * (gamma - 1)))
    else:
        expansion_ratio = 1.0
    
    return {
        'theoretical_isp': isp_theoretical,
        'actual_isp': isp_actual,
        'c_star': c_star_actual,
        'c_f': c_f_actual,
        'exit_velocity': v_exit_actual,
        'exit_pressure': p_exit,
        'exit_temperature': T_exit,
        'is_choked': is_choked,
        'pressure_ratio': pressure_ratio,
        'expansion_ratio': expansion_ratio
    }

@st.cache_data
def calculate_constant_isp_heater_power(target_isp, p_chamber, T_chamber, p_ambient, 
                                       gamma, R_specific, mass_flow_rate, 
                                       current_heat_capacity, base_heater_power=0):
    """
    Calculate required heater power to maintain constant specific impulse
    
    This is a simplified model - assumes Isp is primarily temperature dependent
    """
    
    if target_isp <= 0 or p_chamber <= 0 or mass_flow_rate <= 0:
        return base_heater_power
    
    # Target exit velocity from target Isp
    target_v_exit = target_isp * g0
    
    # For choked flow, relate exit velocity to chamber temperature
    # v_exit ≈ sqrt(gamma * R * T_chamber * factor)
    # Solving for required temperature
    gamma_term = gamma / (gamma - 1)
    temp_factor = gamma * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
    
    target_T_chamber = (target_v_exit ** 2) / (R_specific * temp_factor)
    
    # Temperature difference needed
    delta_T = target_T_chamber - T_chamber
    
    # Additional power needed (simplified)
    if current_heat_capacity > 0:
        additional_power = max(0, delta_T * current_heat_capacity * 0.1)  # Conservative factor
        required_power = base_heater_power + additional_power
    else:
        required_power = base_heater_power
    
    # Limit to reasonable range
    return min(max(required_power, 0), 100)  # 0-100W range

@st.cache_data
def calculate_thrust_parameters(p_chamber, T_chamber, p_ambient, gamma, R_specific,
                               nozzle_efficiency=1.0, c_star_efficiency=0.95):
    """Calculate thrust parameters using enhanced Isp calculation"""
    
    isp_data = calculate_enhanced_isp(p_chamber, T_chamber, p_ambient, gamma, 
                                     R_specific, nozzle_efficiency, c_star_efficiency)
    
    return {
        'exit_velocity': isp_data['exit_velocity'],
        'specific_impulse': isp_data['actual_isp'],
        'theoretical_isp': isp_data['theoretical_isp'],
        'thrust_coefficient': isp_data['c_f'],
        'characteristic_velocity': isp_data['c_star'],
        'exit_pressure': isp_data['exit_pressure'],
        'exit_temperature': isp_data['exit_temperature'],
        'is_choked': isp_data['is_choked'],
        'pressure_ratio': isp_data['pressure_ratio'],
        'expansion_ratio': isp_data['expansion_ratio']
    }

@st.cache_data
def calculate_thrust_force(mass_flow_rate, exit_velocity, exit_pressure, ambient_pressure, exit_area):
    """Calculate thrust force using momentum and pressure terms"""
    
    # Momentum thrust
    F_momentum = mass_flow_rate * exit_velocity
    
    # Pressure thrust
    F_pressure = (exit_pressure - ambient_pressure) * exit_area
    
    # Total thrust
    F_total = F_momentum + F_pressure
    
    return F_total, F_momentum, F_pressure

@st.cache_data
def calculate_orifice_area(m_dot, p1, R_specific, T1, k):
    """Calculate required orifice area for choked flow"""
    if p1 <= 0:
        return 0
    denominator = k * (2 / (k + 1)) ** ((k + 1) / (k - 1))
    return (m_dot / p1) * math.sqrt(R_specific * T1 / denominator)

@st.cache_data
def calculate_mass_flow_from_area(A_t, p1, R_specific, T1, k):
    """Calculate mass flow rate from orifice area"""
    if p1 <= 0 or A_t <= 0:
        return 0
    multiplier = k * (2 / (k + 1)) ** ((k + 1) / (k - 1))
    return A_t * p1 * math.sqrt(multiplier / (R_specific * T1))

@st.cache_data
def calculate_enthalpy_vaporization(T):
    """Watson correlation for butane - verify exponent"""
    if T >= T_critical:
        return 0
    # Use validated correlation for butane
    n = 0.38  # Should be verified from literature
    ratio = (T_critical - T) / (T_critical - 272.05)
    return max(h_vap_normal_kg * (ratio ** n), 0)


def simulate_butane_tank_enhanced_isp(initial_temp_c, initial_mass, target_flow_rate,
                                    dt, max_time, tank_mass, tank_material, 
                                    base_heater_power, tank_initial_temp_c, nozzle_exit_area,
                                    ambient_pressure, isp_mode, target_isp, 
                                    nozzle_efficiency, c_star_efficiency):
    """Enhanced simulation with advanced specific impulse calculations"""
    
    # Initialize variables
    T_liquid = initial_temp_c + 273.15  # K
    T_tank = tank_initial_temp_c + 273.15  # K
    m_liquid = initial_mass
    R_specific = R / M_butane
    
    # Tank material properties
    material_props = TANK_MATERIALS[tank_material]
    tank_heat_capacity = material_props["heat_capacity"]  # J/(kg·K)
    
    # Calculate initial orifice area
    P_initial = antoine_vapor_pressure(T_liquid)
    A_orifice = calculate_orifice_area(target_flow_rate, P_initial, R_specific, T_liquid, gamma)
    
    # Initialize result storage
    results = {
        'time': [0],
        'temperature_liquid': [T_liquid - 273.15],
        'temperature_tank': [T_tank - 273.15],
        'temperature_combined': [T_liquid - 273.15],
        'pressure': [P_initial / 1e5],
        'mass': [m_liquid],
        'flow_rate': [0],
        'h_vap': [calculate_enthalpy_vaporization(T_liquid) / 1000],
        'heat_removed_vaporization': [0],
        'heat_added_heater': [base_heater_power],
        'heat_net': [base_heater_power],
        'heater_status': [1.0 if base_heater_power > 0 else 0.0],
        # Enhanced thrust parameters
        'specific_impulse': [0],
        'theoretical_isp': [0],
        'exit_velocity': [0],
        'thrust_force': [0],
        'thrust_coefficient': [0],
        'characteristic_velocity': [0],
        'exit_pressure': [ambient_pressure / 1e5],
        'exit_temperature': [T_liquid - 273.15],
        'momentum_thrust': [0],
        'pressure_thrust': [0],
        'is_choked': [False],
        'pressure_ratio': [1.0],
        'expansion_ratio': [1.0],
        'nozzle_efficiency': [nozzle_efficiency],
        'actual_heater_power': [base_heater_power],
        'isp_error': [0]
    }
    
    t = 0
    
    while (t < max_time and m_liquid > 0.001 and T_liquid > T_triple and T_liquid < T_critical):
        # Calculate vapor pressure
        P_vapor = antoine_vapor_pressure(T_liquid)
        if P_vapor <= 0:
            break
        
        # Calculate actual mass flow rate
        m_dot_actual = calculate_mass_flow_from_area(A_orifice, P_vapor, R_specific, T_liquid, gamma)
        m_dot_actual = min(m_dot_actual, m_liquid / dt)
        
        # Calculate thrust parameters with enhanced Isp
        thrust_params = calculate_thrust_parameters(
            p_chamber=P_vapor,
            T_chamber=T_liquid,
            p_ambient=ambient_pressure,
            gamma=gamma,
            R_specific=R_specific,
            nozzle_efficiency=nozzle_efficiency,
            c_star_efficiency=c_star_efficiency
        )
        
        # Determine heater power based on Isp mode
        if isp_mode == "Constant" and target_isp > 0:
            # Calculate required heater power for constant Isp
            C_total = m_liquid * c_p_liquid + tank_mass * tank_heat_capacity
            required_power = calculate_constant_isp_heater_power(
                target_isp, P_vapor, T_liquid, ambient_pressure,
                gamma, R_specific, m_dot_actual, C_total, base_heater_power
            )
            current_heater_power = required_power
        else:
            # Use base heater power (variable Isp mode)
            current_heater_power = base_heater_power
        
        # Calculate thrust forces
        F_total, F_momentum, F_pressure = calculate_thrust_force(
            mass_flow_rate=m_dot_actual,
            exit_velocity=thrust_params['exit_velocity'],
            exit_pressure=thrust_params['exit_pressure'],
            ambient_pressure=ambient_pressure,
            exit_area=nozzle_exit_area
        )
        
        # Calculate enthalpy of vaporization at current temperature
        h_vap_current = calculate_enthalpy_vaporization(T_liquid)
        
        # Heat removed by vaporization (cooling effect)
        Q_removed_vaporization = m_dot_actual * h_vap_current  # W (J/s)
        
        # Heat added by heater (heating effect)
        Q_added_heater = current_heater_power  # W
        
        # Net heat transfer
        Q_net = Q_added_heater - Q_removed_vaporization  # W
        
        # Combined thermal mass (liquid + tank)
        C_liquid = m_liquid * c_p_liquid  # J/K
        C_tank = tank_mass * tank_heat_capacity  # J/K
        C_total = C_liquid + C_tank  # J/K
        
        # Temperature change (fully insulated system)
        if C_total > 0:
            dT = (Q_net * dt) / C_total  # K
            T_liquid += dT
            T_tank += dT  # Assume thermal equilibrium
        
        # Update mass
        m_liquid -= m_dot_actual * dt
        
        # Calculate Isp error for constant mode
        if isp_mode == "Constant" and target_isp > 0:
            isp_error = target_isp - thrust_params['specific_impulse']
        else:
            isp_error = 0
        
        # Advance time
        t += dt
        
        # Store results
        results['time'].append(t)
        results['temperature_liquid'].append(T_liquid - 273.15)
        results['temperature_tank'].append(T_tank - 273.15)
        results['temperature_combined'].append(T_liquid - 273.15)
        results['pressure'].append(P_vapor / 1e5)
        results['mass'].append(m_liquid)
        results['flow_rate'].append(m_dot_actual * 1e6)  # μg/s
        results['h_vap'].append(h_vap_current / 1000)  # kJ/kg
        results['heat_removed_vaporization'].append(Q_removed_vaporization)
        results['heat_added_heater'].append(Q_added_heater)
        results['heat_net'].append(Q_net)
        results['heater_status'].append(1.0 if current_heater_power > 0 else 0.0)
        
        # Enhanced thrust results
        results['specific_impulse'].append(thrust_params['specific_impulse'])
        results['theoretical_isp'].append(thrust_params['theoretical_isp'])
        results['exit_velocity'].append(thrust_params['exit_velocity'])
        results['thrust_force'].append(F_total * 1000)  # Convert to mN
        results['thrust_coefficient'].append(thrust_params['thrust_coefficient'])
        results['characteristic_velocity'].append(thrust_params['characteristic_velocity'])
        results['exit_pressure'].append(thrust_params['exit_pressure'] / 1e5)  # bar
        results['exit_temperature'].append(thrust_params['exit_temperature'] - 273.15)  # °C
        results['momentum_thrust'].append(F_momentum * 1000)  # mN
        results['pressure_thrust'].append(F_pressure * 1000)  # mN
        results['is_choked'].append(thrust_params['is_choked'])
        results['pressure_ratio'].append(thrust_params['pressure_ratio'])
        results['expansion_ratio'].append(thrust_params['expansion_ratio'])
        results['nozzle_efficiency'].append(nozzle_efficiency)
        results['actual_heater_power'].append(current_heater_power)
        results['isp_error'].append(isp_error)
        
        # Safety checks
        if T_liquid < T_triple or T_liquid > T_critical:
            break
    
    return {k: np.array(v) for k, v in results.items()}

def create_enhanced_plots_with_isp(results, isp_mode, target_isp):
    """Create enhanced interactive Plotly plots with advanced Isp analysis"""
    
    # Create subplots with 4x3 layout
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=('Temperature Evolution', 'Pressure Evolution', 'Mass Depletion',
                       'Specific Impulse Analysis', 'Thrust Force Components', 'Exit Velocity & Efficiency',
                       'Heat Transfer Analysis', 'Flow Characteristics', 'Isp Performance Metrics',
                       'Nozzle Performance', 'System Control', 'Advanced Diagnostics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    time_hours = results['time'] / 3600
    
    # Row 1: Basic parameters
    fig.add_trace(go.Scatter(x=time_hours, y=results['temperature_combined'], 
                            name='System Temp', line=dict(color='blue', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=time_hours, y=results['pressure'], 
                            name='Pressure', line=dict(color='green', width=2)), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=time_hours, y=results['mass'], 
                            name='Mass', line=dict(color='red', width=2)), row=1, col=3)
    
    # Row 2: Enhanced Isp analysis
    fig.add_trace(go.Scatter(x=time_hours, y=results['specific_impulse'], 
                            name='Actual Isp', line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_hours, y=results['theoretical_isp'], 
                            name='Theoretical Isp', line=dict(color='purple', width=1, dash='dash')), row=2, col=1)
    
    if isp_mode == "Constant" and target_isp > 0:
        fig.add_trace(go.Scatter(x=time_hours, y=[target_isp] * len(time_hours), 
                                name='Target Isp', line=dict(color='red', width=2, dash='dot')), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=time_hours, y=results['thrust_force'], 
                            name='Total Thrust', line=dict(color='orange', width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=time_hours, y=results['momentum_thrust'], 
                            name='Momentum', line=dict(color='green', width=1)), row=2, col=2)
    fig.add_trace(go.Scatter(x=time_hours, y=results['pressure_thrust'], 
                            name='Pressure', line=dict(color='red', width=1)), row=2, col=2)
    
    fig.add_trace(go.Scatter(x=time_hours, y=results['exit_velocity'], 
                            name='Exit Velocity', line=dict(color='brown', width=2)), row=2, col=3)
    fig.add_trace(go.Scatter(x=time_hours, y=results['nozzle_efficiency'] * np.max(results['exit_velocity']), 
                            name='Nozzle Eff. (scaled)', line=dict(color='cyan', width=1)), row=2, col=3)
    
    # Row 3: Heat transfer and flow
    fig.add_trace(go.Scatter(x=time_hours, y=results['heat_added_heater'], 
                            name='Heat Added', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_hours, y=-results['heat_removed_vaporization'], 
                            name='Heat Removed', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_hours, y=results['heat_net'], 
                            name='Net Heat', line=dict(color='black', width=3)), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=time_hours, y=results['flow_rate'], 
                            name='Flow Rate', line=dict(color='cyan', width=2)), row=3, col=2)
    
    # Isp performance metrics
    if isp_mode == "Constant":
        fig.add_trace(go.Scatter(x=time_hours, y=results['isp_error'], 
                                name='Isp Error', line=dict(color='red', width=2)), row=3, col=3)
    else:
        fig.add_trace(go.Scatter(x=time_hours, y=results['specific_impulse'], 
                                name='Variable Isp', line=dict(color='purple', width=2)), row=3, col=3)
    
    # Row 4: Advanced diagnostics
    fig.add_trace(go.Scatter(x=time_hours, y=results['thrust_coefficient'], 
                            name='Thrust Coeff', line=dict(color='darkgreen', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=time_hours, y=results['expansion_ratio'], 
                            name='Expansion Ratio', line=dict(color='orange', width=1)), row=4, col=1)
    
    # Control system
    fig.add_trace(go.Scatter(x=time_hours, y=results['actual_heater_power'], 
                            name='Actual Power', line=dict(color='red', width=2)), row=4, col=2)
    
    # Choked flow indicator
    choked_indicator = [1 if choked else 0 for choked in results['is_choked']]
    fig.add_trace(go.Scatter(x=time_hours, y=choked_indicator, 
                            name='Choked Flow', line=dict(color='purple', width=2)), row=4, col=3)
    fig.add_trace(go.Scatter(x=time_hours, y=results['pressure_ratio'], 
                            name='Pressure Ratio', line=dict(color='blue', width=1)), row=4, col=3)
    
    # Update axes labels
    axes_labels = [
        ('Temperature (°C)', 'Pressure (bar)', 'Mass (kg)'),
        ('Specific Impulse (s)', 'Thrust Force (mN)', 'Exit Velocity (m/s)'),
        ('Heat Transfer (W)', 'Flow Rate (μg/s)', 'Isp Error (s)' if isp_mode == "Constant" else 'Specific Impulse (s)'),
        ('Thrust Coefficient', 'Heater Power (W)', 'Flow State')
    ]
    
    # Apply axis labels 
        # Apply axis labels to each subplot
    for i, row_labels in enumerate(axes_labels, start=1):
        for j, y_label in enumerate(row_labels, start=1):
            fig.update_yaxes(title_text=y_label, row=i, col=j)

    # Update x-axis labels for all plots
    for i in range(1, 5):  # 4 rows
        for j in range(1, 4):  # 3 columns
            fig.update_xaxes(title_text="Time (h)", row=i, col=j)

    # Final layout adjustments
    fig.update_layout(
        height=1600,
        width=1400,
        title_text="Butane Tank Simulation Diagnostics",
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig
# Sidebar for input parameters
with st.sidebar:
    st.header("🎛️ Simulation Parameters")
    
    # Tank Configuration
    st.subheader("Tank Configuration")
    initial_temp_c = st.slider("Initial Liquid Temperature (°C)", -20, 50, 20, 1)
    tank_initial_temp_c = st.slider("Initial Tank Temperature (°C)", -20, 50, 20, 1)
    initial_mass = st.slider("Initial Butane Mass (kg)", 0.001, 1.0, 0.1, 0.001)
    tank_mass = st.slider("Tank Mass (kg)", 0.01, 0.5, 0.05, 0.01)
    tank_material = st.selectbox("Tank Material", list(TANK_MATERIALS.keys()))
    
    # Flow and Nozzle Parameters
    st.subheader("Flow & Nozzle Configuration")
    target_flow_rate = st.slider("Target Flow Rate (μg/s)", 1, 100, 10, 1) * 1e-6  # Convert to kg/s
    nozzle_exit_area = st.slider("Nozzle Exit Area (mm²)", 0.1, 10.0, 1.0, 0.1) * 1e-6  # Convert to m²
    ambient_pressure = st.slider("Ambient Pressure (bar)", 0.1, 2.0, 1.013, 0.001) * 1e5  # Convert to Pa
    
    # Efficiency Parameters
    st.subheader("Efficiency Parameters")
    nozzle_efficiency = st.slider("Nozzle Efficiency", 0.5, 1.0, 0.85, 0.01)
    c_star_efficiency = st.slider("Combustion Efficiency", 0.7, 1.0, 0.95, 0.01)
    
    # Heater Configuration
    st.subheader("Heater Configuration")
    base_heater_power = st.slider("Base Heater Power (W)", 0, 50, 5, 1)
    
    # Isp Mode Selection
    st.subheader("Specific Impulse Control")
    isp_mode = st.selectbox("Isp Mode", ["Variable", "Constant"])
    
    if isp_mode == "Constant":
        target_isp = st.slider("Target Specific Impulse (s)", 10, 200, 80, 1)
        st.info("💡 In constant Isp mode, heater power will automatically adjust to maintain target Isp")
    else:
        target_isp = 0
        st.info("💡 In variable Isp mode, Isp will vary with temperature and pressure")
    
    # Simulation Parameters
    st.subheader("Simulation Settings")
    max_time_hours = st.slider("Simulation Duration (hours)", 0.1, 24.0, 2.0, 0.1)
    time_step = st.selectbox("Time Step (seconds)", [0.1, 0.5, 1.0, 5.0], index=2)
    
    # Convert time parameters
    max_time = max_time_hours * 3600  # Convert to seconds
    dt = time_step

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🚀 Simulation Controls")
    
    if st.button("🔥 Run Enhanced Simulation", type="primary"):
        with st.spinner("Running enhanced simulation with Isp calculations..."):
            try:
                # Run the enhanced simulation
                results = simulate_butane_tank_enhanced_isp(
                    initial_temp_c=initial_temp_c,
                    initial_mass=initial_mass,
                    target_flow_rate=target_flow_rate,
                    dt=dt,
                    max_time=max_time,
                    tank_mass=tank_mass,
                    tank_material=tank_material,
                    base_heater_power=base_heater_power,
                    tank_initial_temp_c=tank_initial_temp_c,
                    nozzle_exit_area=nozzle_exit_area,
                    ambient_pressure=ambient_pressure,
                    isp_mode=isp_mode,
                    target_isp=target_isp,
                    nozzle_efficiency=nozzle_efficiency,
                    c_star_efficiency=c_star_efficiency
                )
                
                st.session_state.simulation_results = results
                st.session_state.simulation_params = {
                    'isp_mode': isp_mode,
                    'target_isp': target_isp,
                    'nozzle_efficiency': nozzle_efficiency,
                    'c_star_efficiency': c_star_efficiency
                }
                
                st.success("✅ Enhanced simulation completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Simulation failed: {str(e)}")

with col2:
    st.subheader("📊 Quick Stats")
    
    if 'simulation_results' in st.session_state:
        results = st.session_state.simulation_results
        
        # Calculate final statistics
        final_mass = results['mass'][-1]
        initial_mass_sim = results['mass'][0]
        mass_consumed = initial_mass_sim - final_mass
        consumption_rate = mass_consumed / (results['time'][-1] / 3600)  # kg/h
        
        avg_isp = np.mean(results['specific_impulse'][1:])  # Skip first zero
        max_thrust = np.max(results['thrust_force'])
        avg_thrust = np.mean(results['thrust_force'][1:])
        
        # Display metrics
        st.markdown(f"""
        <div class="metric-card">
            <strong>Mass Consumed:</strong> {mass_consumed*1000:.1f} g<br>
            <strong>Consumption Rate:</strong> {consumption_rate*1000:.2f} g/h<br>
            <strong>Avg Specific Impulse:</strong> {avg_isp:.1f} s<br>
            <strong>Max Thrust:</strong> {max_thrust:.2f} mN<br>
            <strong>Avg Thrust:</strong> {avg_thrust:.2f} mN
        </div>
        """, unsafe_allow_html=True)
        
        # Isp mode indicator
        if st.session_state.simulation_params['isp_mode'] == "Constant":
            isp_error_rms = np.sqrt(np.mean(results['isp_error'][1:]**2))
            st.markdown(f"""
            <div class="isp-box">
                <strong>🎯 Constant Isp Mode</strong><br>
                Target: {st.session_state.simulation_params['target_isp']} s<br>
                RMS Error: {isp_error_rms:.2f} s
            </div>
            """, unsafe_allow_html=True)
        else:
            isp_variation = np.std(results['specific_impulse'][1:])
            st.markdown(f"""
            <div class="thrust-box">
                <strong>📈 Variable Isp Mode</strong><br>
                Avg Isp: {avg_isp:.1f} s<br>
                Std Dev: {isp_variation:.2f} s
            </div>
            """, unsafe_allow_html=True)

# Display results if available
if 'simulation_results' in st.session_state:
    results = st.session_state.simulation_results
    params = st.session_state.simulation_params
    
    # Create and display enhanced plots
    st.subheader("📈 Enhanced Simulation Results")
    
    fig = create_enhanced_plots_with_isp(results, params['isp_mode'], params['target_isp'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Analysis Section
    st.subheader("🔬 Performance Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Specific Impulse Performance")
        
        # Isp statistics
        valid_isp = results['specific_impulse'][1:]  # Skip first zero
        if len(valid_isp) > 0:
            avg_isp = np.mean(valid_isp)
            min_isp = np.min(valid_isp)
            max_isp = np.max(valid_isp)
            isp_efficiency = np.mean(results['specific_impulse'][1:] / results['theoretical_isp'][1:]) * 100
            
            st.markdown(f"""
            <div class="isp-box">
                <strong>Average Isp:</strong> {avg_isp:.1f} s<br>
                <strong>Min Isp:</strong> {min_isp:.1f} s<br>
                <strong>Max Isp:</strong> {max_isp:.1f} s<br>
                <strong>Isp Efficiency:</strong> {isp_efficiency:.1f}%
            </div>
            """, unsafe_allow_html=True)
            
            if params['isp_mode'] == "Constant":
                target = params['target_isp']
                avg_error = np.mean(np.abs(results['isp_error'][1:]))
                max_error = np.max(np.abs(results['isp_error'][1:]))
                
                st.markdown(f"""
                <div class="success-box">
                    <strong>🎯 Constant Isp Control:</strong><br>
                    Target: {target} s<br>
                    Avg Error: ±{avg_error:.2f} s<br>
                    Max Error: ±{max_error:.2f} s
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🚀 Thrust Performance")
        
        # Thrust statistics
        valid_thrust = results['thrust_force'][1:]
        if len(valid_thrust) > 0:
            avg_thrust = np.mean(valid_thrust)
            max_thrust = np.max(valid_thrust)
            min_thrust = np.min(valid_thrust)
            
            # Calculate thrust-to-weight ratio (assuming 1g = 9.81 mN)
            twr = avg_thrust / (initial_mass * 9810)  # Thrust-to-weight ratio
            
            st.markdown(f"""
            <div class="thrust-box">
                <strong>Average Thrust:</strong> {avg_thrust:.2f} mN<br>
                <strong>Max Thrust:</strong> {max_thrust:.2f} mN<br>
                <strong>Min Thrust:</strong> {min_thrust:.2f} mN<br>
                <strong>Thrust-to-Weight:</strong> {twr:.4f}
            </div>
            """, unsafe_allow_html=True)
            
            # Flow regime analysis
            choked_percentage = np.mean(results['is_choked'][1:]) * 100
            avg_expansion_ratio = np.mean(results['expansion_ratio'][1:])
            
            st.markdown(f"""
            <div class="insulated-box">
                <strong>🌪️ Flow Analysis:</strong><br>
                Choked Flow: {choked_percentage:.1f}% of time<br>
                Avg Expansion Ratio: {avg_expansion_ratio:.2f}
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### ⚡ System Efficiency")
        
        # System efficiency metrics
        avg_nozzle_eff = params['nozzle_efficiency'] * 100
        avg_cstar_eff = params['c_star_efficiency'] * 100
        
        # Calculate overall system efficiency
        theoretical_exhaust_velocity = np.mean(results['exit_velocity'][1:] / (params['nozzle_efficiency'] * np.sqrt(params['c_star_efficiency'])))
        actual_exhaust_velocity = np.mean(results['exit_velocity'][1:])
        overall_efficiency = (actual_exhaust_velocity / theoretical_exhaust_velocity) * 100 if theoretical_exhaust_velocity > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Nozzle Efficiency:</strong> {avg_nozzle_eff:.1f}%<br>
            <strong>C* Efficiency:</strong> {avg_cstar_eff:.1f}%<br>
            <strong>Overall Efficiency:</strong> {overall_efficiency:.1f}%<br>
            <strong>Avg Exit Velocity:</strong> {actual_exhaust_velocity:.1f} m/s
        </div>
        """, unsafe_allow_html=True)
        
        # Power consumption analysis
        if params['isp_mode'] == "Constant":
            avg_power = np.mean(results['actual_heater_power'][1:])
            max_power = np.max(results['actual_heater_power'][1:])
            power_variation = np.std(results['actual_heater_power'][1:])
            
            st.markdown(f"""
            <div class="warning-box">
                <strong>🔋 Power Control:</strong><br>
                Avg Power: {avg_power:.1f} W<br>
                Max Power: {max_power:.1f} W<br>
                Power Variation: ±{power_variation:.1f} W
            </div>
            """, unsafe_allow_html=True)

    # Data Export Section
    st.subheader("📥 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download Simulation Data (CSV)"):
            # Create DataFrame
            df = pd.DataFrame({
                'Time_hours': results['time'] / 3600,
                'Temperature_C': results['temperature_combined'],
                'Pressure_bar': results['pressure'],
                'Mass_kg': results['mass'],
                'Flow_rate_ug_s': results['flow_rate'],
                'Specific_Impulse_s': results['specific_impulse'],
                'Theoretical_Isp_s': results['theoretical_isp'],
                'Thrust_mN': results['thrust_force'],
                'Exit_Velocity_m_s': results['exit_velocity'],
                'Heater_Power_W': results['actual_heater_power'],
                'Nozzle_Efficiency': results['nozzle_efficiency'],
                'Is_Choked': results['is_choked'],
                'Isp_Error_s': results['isp_error']
            })
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name="butane_simulation_enhanced.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Download Performance Summary"):
            # Create performance summary
            summary = {
                'Simulation Parameters': {
                    'Initial Mass (kg)': initial_mass,
                    'Initial Temperature (°C)': initial_temp_c,
                    'Tank Material': tank_material,
                    'Isp Mode': params['isp_mode'],
                    'Target Isp (s)': params['target_isp'] if params['isp_mode'] == "Constant" else "Variable",
                    'Nozzle Efficiency': params['nozzle_efficiency'],
                    'Simulation Duration (h)': results['time'][-1] / 3600
                },
                'Performance Metrics': {
                    'Average Isp (s)': np.mean(results['specific_impulse'][1:]),
                    'Average Thrust (mN)': np.mean(results['thrust_force'][1:]),
                    'Max Thrust (mN)': np.max(results['thrust_force']),
                    'Mass Consumed (g)': (results['mass'][0] - results['mass'][-1]) * 1000,
                    'Average Power (W)': np.mean(results['actual_heater_power'][1:]),
                    'Choked Flow (%)': np.mean(results['is_choked'][1:]) * 100
                }
            }
            
            summary_text = "# Butane Tank Simulation Performance Summary\n\n"
            for section, values in summary.items():
                summary_text += f"## {section}\n"
                for key, value in values.items():
                    if isinstance(value, float):
                        summary_text += f"- {key}: {value:.3f}\n"
                    else:
                        summary_text += f"- {key}: {value}\n"
                summary_text += "\n"
            
            st.download_button(
                label="📋 Download Summary",
                data=summary_text,
                file_name="simulation_summary.md",
                mime="text/markdown"
            )

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <h4>🔬 Enhanced Butane Tank Thermal Simulation</h4>
    <p>Advanced specific impulse calculations with constant Isp control capability</p>
    <p><em>Features: Variable/Constant Isp modes, Nozzle efficiency modeling, Advanced thrust calculations</em></p>
</div>
""", unsafe_allow_html=True)