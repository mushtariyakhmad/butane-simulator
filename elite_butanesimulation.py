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
    page_title="Butane Tank Thermal Simulation", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Ultra-Modern CSS with Glass Morphism and Advanced Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Advanced CSS Variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        --secondary-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --accent-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        --glass-bg: rgba(255, 255, 255, 0.25);
        --glass-border: rgba(255, 255, 255, 0.18);
        --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.37);
        --shadow-heavy: 0 15px 35px rgba(31, 38, 135, 0.5);
        --border-radius: 20px;
        --animation-speed: 0.4s;
    }
    
    /* Global Reset and Base Styling */
    * {
        transition: all var(--animation-speed) cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .main .block-container {
        padding-top: 1rem;
        max-width: 1600px;
        margin: 0 auto;
    }
    
    /* Revolutionary Header with Particles Effect */
    .hero-header {
        background: var(--primary-gradient);
        padding: 4rem 2rem;
        border-radius: 30px;
        margin-bottom: 3rem;
        box-shadow: var(--shadow-heavy);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
        animation: headerFloat 6s ease-in-out infinite alternate;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: particleMove 20s linear infinite;
    }
    
    .hero-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes headerFloat {
        0% { 
            transform: translateY(0) scale(1);
            box-shadow: var(--shadow-heavy);
        }
        100% { 
            transform: translateY(-5px) scale(1.002);
            box-shadow: 0 25px 50px rgba(31, 38, 135, 0.6);
        }
    }
    
    @keyframes particleMove {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-50px, -50px) rotate(360deg); }
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0; }
        50% { opacity: 1; }
    }
    
    .hero-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: clamp(2rem, 5vw, 4rem);
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        text-shadow: 2px 2px 20px rgba(0,0,0,0.5);
        position: relative;
        z-index: 2;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        position: relative;
        z-index: 2;
        font-weight: 400;
        font-size: 1.2rem;
        opacity: 0.95;
        margin-top: 1rem;
        text-align: center;
        color: rgba(255,255,255,0.9);
    }
    
    /* Glass Morphism Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid var(--glass-border);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-light);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(31, 38, 135, 0.4);
        border-color: rgba(255,255,255,0.3);
    }
    
    /* Neon Performance Cards */
    .neon-card {
        background: linear-gradient(145deg, rgba(0,0,0,0.1), rgba(255,255,255,0.1));
        backdrop-filter: blur(20px);
        border: 2px solid transparent;
        border-radius: 25px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
        animation: cardPulse 4s ease-in-out infinite;
    }
    
    .neon-card::before {
        content: '';
        position: absolute;
        inset: 0;
        padding: 2px;
        background: var(--primary-gradient);
        border-radius: inherit;
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: xor;
        -webkit-mask-composite: xor;
    }
    
    .neon-card:hover::before {
        background: var(--accent-gradient);
        animation: neonGlow 1s ease-in-out;
    }
    
    @keyframes cardPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 0 40px rgba(118, 75, 162, 0.5), 0 0 60px rgba(240, 147, 251, 0.3); }
    }
    
    @keyframes neonGlow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.2) saturate(1.3); }
    }
    
    /* Specialized Performance Boxes */
    .thrust-performance {
        background: linear-gradient(145deg, #fff5e6 0%, #ffe0b3 50%, #ffcc80 100%);
        border: 3px solid transparent;
        border-radius: 25px;
        padding: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(255, 152, 0, 0.3);
    }
    
    .thrust-performance::before {
        content: '';
        position: absolute;
        inset: 0;
        padding: 3px;
        background: linear-gradient(45deg, #ff9800, #ff5722, #ff9800);
        border-radius: inherit;
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: xor;
        -webkit-mask-composite: xor;
        animation: borderRotate 3s linear infinite;
    }
    
    @keyframes borderRotate {
        0% { background: linear-gradient(45deg, #ff9800, #ff5722, #ff9800); }
        33% { background: linear-gradient(45deg, #ff5722, #ff9800, #ff5722); }
        66% { background: linear-gradient(45deg, #ff9800, #ff5722, #ff9800); }
        100% { background: linear-gradient(45deg, #ff5722, #ff9800, #ff5722); }
    }
    
    .thermal-performance {
        background: var(--secondary-gradient);
        color: white;
        border-radius: 25px;
        padding: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(79, 172, 254, 0.4);
    }
    
    .thermal-performance::after {
        content: '🌡️';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 3rem;
        opacity: 0.3;
        animation: thermalPulse 2s ease-in-out infinite;
    }
    
    @keyframes thermalPulse {
        0%, 100% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.1) rotate(5deg); }
    }
    
    .efficiency-performance {
        background: var(--accent-gradient);
        color: white;
        border-radius: 25px;
        padding: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(67, 233, 123, 0.4);
    }
    
    .efficiency-performance::after {
        content: '⚡';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 3rem;
        opacity: 0.3;
        animation: electricPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes electricPulse {
        0%, 100% { transform: scale(1); filter: brightness(1); }
        50% { transform: scale(1.2); filter: brightness(1.3); }
    }
    
    /* Revolutionary Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.1) 0%, rgba(255,255,255,0.95) 30%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(20px);
        border-right: 3px solid var(--glass-border);
    }
    
    /* Advanced Control Panel */
    .control-nexus {
        background: var(--glass-bg);
        backdrop-filter: blur(30px);
        border: 2px solid var(--glass-border);
        border-radius: 30px;
        padding: 3rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-heavy);
    }
    
    .control-nexus::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        animation: conicRotate 10s linear infinite;
    }
    
    @keyframes conicRotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .control-nexus h2 {
        position: relative;
        z-index: 2;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        margin-top: 0;
    }
    
    /* Quantum Buttons */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 1.2rem 3rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    /* Status Indicators with Glow */
    .status-excellent { 
        color: #00ff88; 
        font-weight: 700; 
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        animation: statusGlow 2s ease-in-out infinite alternate;
    }
    
    .status-good { 
        color: #28a745; 
        font-weight: 700;
        text-shadow: 0 0 8px rgba(40, 167, 69, 0.4);
    }
    
    .status-warning { 
        color: #ffc107; 
        font-weight: 700;
        text-shadow: 0 0 8px rgba(255, 193, 7, 0.4);
        animation: warningPulse 1.5s ease-in-out infinite;
    }
    
    .status-danger { 
        color: #dc3545; 
        font-weight: 700;
        text-shadow: 0 0 10px rgba(220, 53, 69, 0.5);
        animation: dangerFlash 1s ease-in-out infinite alternate;
    }
    
    @keyframes statusGlow {
        0% { text-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }
        100% { text-shadow: 0 0 20px rgba(0, 255, 136, 0.8), 0 0 30px rgba(0, 255, 136, 0.3); }
    }
    
    @keyframes warningPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes dangerFlash {
        0% { text-shadow: 0 0 10px rgba(220, 53, 69, 0.5); }
        100% { text-shadow: 0 0 20px rgba(220, 53, 69, 0.9), 0 0 30px rgba(220, 53, 69, 0.4); }
    }
    
    /* Holographic Data Display */
    .data-hologram {
        background: linear-gradient(145deg, rgba(0,20,40,0.9), rgba(0,40,80,0.8));
        border: 2px solid rgba(0, 255, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        color: #00ffff;
        font-family: 'JetBrains Mono', monospace;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.3);
    }
    
    .data-hologram::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.1), transparent);
        animation: scanLine 3s linear infinite;
    }
    
    @keyframes scanLine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Formula Display */
    .formula-showcase {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 2px solid #0f3460;
        border-radius: 20px;
        padding: 2rem;
        color: #e94560;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin: 1.5rem 0;
    }
    
    .formula-showcase::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 48%, rgba(233, 69, 96, 0.1) 49%, rgba(233, 69, 96, 0.1) 51%, transparent 52%);
        animation: formulaGrid 4s linear infinite;
    }
    
    @keyframes formulaGrid {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Enhanced Charts Container */
    .chart-nexus {
        background: var(--glass-bg);
        backdrop-filter: blur(25px);
        border: 2px solid var(--glass-border);
        border-radius: 30px;
        padding: 3rem;
        margin: 3rem 0;
        box-shadow: var(--shadow-heavy);
        position: relative;
        overflow: hidden;
    }
    
    .chart-nexus::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: var(--primary-gradient);
        border-radius: inherit;
        z-index: -1;
        opacity: 0.7;
    }
    
    /* Advanced Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 1.3rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Loading Animations */
    .quantum-loader {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: quantumSpin 1s ease-in-out infinite;
    }
    
    @keyframes quantumSpin {
        0% { transform: rotate(0deg) scale(1); }
        50% { transform: rotate(180deg) scale(1.1); }
        100% { transform: rotate(360deg) scale(1); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-header {
            padding: 2rem 1rem;
        }
        .hero-header h1 {
            font-size: 2.5rem;
        }
        .glass-card, .neon-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-gradient);
    }
    
    /* Footer Enhancement */
    .quantum-footer {
        background: var(--dark-gradient);
        color: white;
        padding: 4rem 2rem;
        border-radius: 30px;
        margin-top: 4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .quantum-footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="20" cy="20" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="80" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="40" cy="60" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="60" cy="40" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Butane Tank Thermal Simulation </h1>
    <p style="text-align: center; color: white; margin: 0;">
        Simple constant Isp model: Thrust = Isp × Mass Flow × g₀
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
def calculate_simple_thrust(mass_flow_rate, isp_constant):
    """
    Simple thrust calculation: Thrust = Isp × mass_flow_rate × g₀
    
    Parameters:
    - mass_flow_rate: kg/s
    - isp_constant: specific impulse in seconds (constant)
    
    Returns:
    - thrust_force: thrust in Newtons
    """
    return isp_constant * mass_flow_rate * g0

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
    """Watson correlation for butane"""
    if T >= T_critical:
        return 0
    n = 0.38
    ratio = (T_critical - T) / (T_critical - 272.05)
    return max(h_vap_normal_kg * (ratio ** n), 0)

def simulate_butane_tank_simple(initial_temp_c, initial_mass, target_flow_rate,
                               dt, max_time, tank_mass, tank_material, 
                               heater_power, tank_initial_temp_c, isp_constant):
    """Simulation with constant Isp"""
    
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
        'heat_added_heater': [heater_power],
        'heat_net': [heater_power],
        'heater_status': [1.0 if heater_power > 0 else 0.0],
        # Simplified thrust parameters
        'specific_impulse': [isp_constant],
        'thrust_force': [0],  # Will be in Newtons
        'thrust_force_mN': [0]  # Will be in milliNewtons for display
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
        
        # Simple thrust calculation: Thrust = Isp × mass_flow × g₀
        thrust_N = calculate_simple_thrust(m_dot_actual, isp_constant)
        thrust_mN = thrust_N * 1000  # Convert to milliNewtons
        
        # Calculate enthalpy of vaporization at current temperature
        h_vap_current = calculate_enthalpy_vaporization(T_liquid)
        
        # Heat removed by vaporization (cooling effect)
        Q_removed_vaporization = m_dot_actual * h_vap_current  # W (J/s)
        
        # Heat added by heater (heating effect)
        Q_added_heater = heater_power  # W
        
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
        results['heater_status'].append(1.0 if heater_power > 0 else 0.0)
        
        # Simplified thrust results
        results['specific_impulse'].append(isp_constant)
        results['thrust_force'].append(thrust_N)
        results['thrust_force_mN'].append(thrust_mN)
        
        # Safety checks
        if T_liquid < T_triple or T_liquid > T_critical:
            break
    
    return {k: np.array(v) for k, v in results.items()}

def create_simplified_plots(results, isp_constant):
    """Create interactive Plotly plots"""
    
    # Create subplots with 3x2 layout
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Temperature Evolution', 'Pressure Evolution',
                       'Mass Depletion', 'Flow Rate',
                       'Heat Transfer Analysis', 'Thrust Performance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    time_hours = results['time'] / 3600
    
    # Row 1: Temperature and Pressure
    fig.add_trace(go.Scatter(x=time_hours, y=results['temperature_combined'], 
                            name='System Temp', line=dict(color='blue', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=time_hours, y=results['pressure'], 
                            name='Pressure', line=dict(color='green', width=2)), row=1, col=2)
    
    # Row 2: Mass and Flow Rate
    fig.add_trace(go.Scatter(x=time_hours, y=results['mass'], 
                            name='Mass', line=dict(color='red', width=2)), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=time_hours, y=results['flow_rate'], 
                            name='Flow Rate', line=dict(color='cyan', width=2)), row=2, col=2)
    
    # Row 3: Heat Transfer and Thrust
    fig.add_trace(go.Scatter(x=time_hours, y=results['heat_added_heater'], 
                            name='Heat Added', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_hours, y=-results['heat_removed_vaporization'], 
                            name='Heat Removed', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_hours, y=results['heat_net'], 
                            name='Net Heat', line=dict(color='black', width=3)), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=time_hours, y=results['thrust_force_mN'], 
                            name='Thrust Force', line=dict(color='purple', width=2)), row=3, col=2)
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (h)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    
    fig.update_xaxes(title_text="Time (h)", row=1, col=2)
    fig.update_yaxes(title_text="Pressure (bar)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time (h)", row=2, col=1)
    fig.update_yaxes(title_text="Mass (kg)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (h)", row=2, col=2)
    fig.update_yaxes(title_text="Flow Rate (μg/s)", row=2, col=2)
    
    fig.update_xaxes(title_text="Time (h)", row=3, col=1)
    fig.update_yaxes(title_text="Heat Transfer (W)", row=3, col=1)
    
    fig.update_xaxes(title_text="Time (h)", row=3, col=2)
    fig.update_yaxes(title_text="Thrust Force (mN)", row=3, col=2)
    
    # Final layout adjustments
    fig.update_layout(
        height=1200,
        width=1200,
        title_text=f"Butane Tank Simulation - Constant Isp = {isp_constant} s",
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
    initial_temp_c = st.number_input(
        "Initial Liquid Temperature (°C)", 
        min_value=-20.0, 
        max_value=50.0, 
        value=20.0, 
        step=None  # Allows any value (-3.7 or 42.123)
    )
    tank_initial_temp_c = st.number_input(
        "Initial Tank Temperature (°C)", 
        min_value=-20.0, 
        max_value=50.0, 
        value=20.0, 
        step=None
    )
    initial_mass = st.number_input(
        "Initial Butane Mass (kg)", 
        min_value=0.001, 
        max_value=1.0, 
        value=0.1, 
        step=None, 
        format="%.6f"  # Show 6 decimal places
    )
    tank_mass = st.number_input(
        "Tank Mass (kg)", 
        min_value=0.01, 
        max_value=0.5, 
        value=0.05, 
        step=None
    )
    tank_material = st.selectbox("Tank Material", list(TANK_MATERIALS.keys()))
    
    # Flow Parameters
    st.subheader("Flow Configuration")
    target_flow_rate = st.number_input(
    "Target Flow Rate (kg/s)", 
    min_value=1e-8, 
    max_value=1e-3, 
    value=5.5e-5,  # default value
    step=1e-6,
    format="%.2e"  # Scientific notation format
    )
    # Simplified Isp Parameter
    st.subheader("🚀 Specific Impulse (Constant)")
    isp_constant = st.number_input(
        "Constant Isp (s)", 
        min_value=10.0, 
        max_value=200.0, 
        value=80.0, 
        step=None
    )
    
    st.markdown(f"""
    <div class="isp-box">
        <strong>📐 Simple Formula:</strong><br>
        Thrust = {isp_constant} s × Mass Flow × 9.807 m/s²<br>
        <em>Thrust directly proportional to flow rate</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Heater Configuration
    st.subheader("Heater Configuration")
    heater_power = st.number_input(
        "Heater Power (W)", 
        min_value=0.0, 
        max_value=50.0, 
        value=5.0, 
        step=None
    )
    
    # Simulation Parameters
    st.subheader("Simulation Settings")
    max_time_hours = st.number_input(
        "Simulation Duration (hours)", 
        min_value=0.1, 
        max_value=24.0, 
        value=2.0, 
        step=None
    )
    time_step = st.number_input(
        "Time Step (seconds)", 
        min_value=0.001,  # Very small step allowed
        max_value=10.0, 
        value=1.0, 
        step=None
    )
    # Convert time parameters
    max_time = max_time_hours * 3600  # Convert to seconds
    dt = time_step

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(" Simulation Controls")
    
    if st.button("🔥 Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                # Run the simplified simulation
                results = simulate_butane_tank_simple(
                    initial_temp_c=initial_temp_c,
                    initial_mass=initial_mass,
                    target_flow_rate=target_flow_rate,
                    dt=dt,
                    max_time=max_time,
                    tank_mass=tank_mass,
                    tank_material=tank_material,
                    heater_power=heater_power,
                    tank_initial_temp_c=tank_initial_temp_c,
                    isp_constant=isp_constant
                )
                
                st.session_state.simulation_results = results
                st.session_state.isp_constant = isp_constant
                
                st.success("✅ Simulation completed successfully!")
                
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
        
        avg_flow_rate = np.mean(results['flow_rate'][1:]) * 1e-6  # kg/s
        avg_thrust_mN = np.mean(results['thrust_force_mN'][1:])
        max_thrust_mN = np.max(results['thrust_force_mN'])
        
        # Display metrics
        st.markdown(f"""
        <div class="metric-card">
            <strong>Mass Consumed:</strong> {mass_consumed*1000:.1f} g<br>
            <strong>Consumption Rate:</strong> {consumption_rate*1000:.2f} g/h<br>
            <strong>Avg Flow Rate:</strong> {avg_flow_rate*1e6:.1f} μg/s<br>
            <strong>Avg Thrust:</strong> {avg_thrust_mN:.2f} mN<br>
            <strong>Max Thrust:</strong> {max_thrust_mN:.2f} mN
        </div>
        """, unsafe_allow_html=True)
        
        # Thrust calculation display
        st.markdown(f"""
        <div class="thrust-box">
            <strong>🧮 Thrust Calculation:</strong><br>
            Isp: {st.session_state.isp_constant} s<br>
            Avg Flow: {avg_flow_rate*1e6:.1f} μg/s<br>
            Formula: {st.session_state.isp_constant} × {avg_flow_rate*1e6:.1f}e-6 × 9.807<br>
            = <strong>{avg_thrust_mN:.2f} mN</strong>
        </div>
        """, unsafe_allow_html=True)

# Display results if available
if 'simulation_results' in st.session_state:
    results = st.session_state.simulation_results
    isp_const = st.session_state.isp_constant
    
    # Create and display plots
    st.subheader("📈 Simulation Results")
    
    fig = create_simplified_plots(results, isp_const)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Analysis Section
    st.subheader("🔬 Performance Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Thrust Performance")
        
        # Thrust statistics
        valid_thrust = results['thrust_force_mN'][1:]
        if len(valid_thrust) > 0:
            avg_thrust_mN = np.mean(valid_thrust)
            max_thrust_mN = np.max(valid_thrust)
            min_thrust_mN = np.min(valid_thrust)
            thrust_variation = np.std(valid_thrust)
            
            st.markdown(f"""
            <div class="thrust-box">
                <strong>Constant Isp:</strong> {isp_const} s<br>
                <strong>Average Thrust:</strong> {avg_thrust_mN:.2f} mN<br>
                <strong>Max Thrust:</strong> {max_thrust_mN:.2f} mN<br>
                <strong>Min Thrust:</strong> {min_thrust_mN:.2f} mN<br>
                <strong>Std Deviation:</strong> {thrust_variation:.2f} mN
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🌡️ Thermal Performance")
        
        # Temperature statistics
        temp_change = results['temperature_combined'][-1] - results['temperature_combined'][0]
        min_temp = np.min(results['temperature_combined'])
        max_temp = np.max(results['temperature_combined'])
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Temperature Change:</strong> {temp_change:.1f} °C<br>
            <strong>Min Temperature:</strong> {min_temp:.1f} °C<br>
            <strong>Max Temperature:</strong> {max_temp:.1f} °C<br>
            <strong>Final Temperature:</strong> {results['temperature_combined'][-1]:.1f} °C
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### ⚡ System Efficiency")
        
        # System metrics
        avg_power = heater_power
        total_energy = avg_power * (results['time'][-1] / 3600)  # Wh
        thrust_time_integral = np.trapz(results['thrust_force_mN'][1:], results['time'][1:]) / 1000  # N⋅s
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Heater Power:</strong> {avg_power:.1f} W<br>
            <strong>Total Energy:</strong> {total_energy:.2f} Wh<br>
            <strong>Total Impulse:</strong> {thrust_time_integral:.3f} N⋅s<br>
            <strong>Impulse/Energy:</strong> {thrust_time_integral/total_energy if total_energy > 0 else 0:.4f} N⋅s/Wh
        </div>
        """, unsafe_allow_html=True)

    # Data Export Section
    st.subheader("📥 Data Export")
    
    if st.button("📊 Download Simulation Data (CSV)"):
        # Create DataFrame
        df = pd.DataFrame({
            'Time_hours': results['time'] / 3600,
            'Temperature_C': results['temperature_combined'],
            'Pressure_bar': results['pressure'],
            'Mass_kg': results['mass'],
            'Flow_rate_ug_s': results['flow_rate'],
            'Specific_Impulse_s': results['specific_impulse'],
            'Thrust_mN': results['thrust_force_mN'],
            'Thrust_N': results['thrust_force'],
            'Heater_Power_W': results['heat_added_heater'],
            'Heat_Removed_W': results['heat_removed_vaporization'],
            'Net_Heat_W': results['heat_net']
        })
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name="butane_simulation.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <h4>🔬 Butane Tank Thermal Simulation</h4>
    <p>Constant specific impulse model: <strong>Thrust = Isp × Mass Flow × g₀</strong></p>
    <p><em>Simple thrust calculations for initial modeling</em></p>
</div>
""", unsafe_allow_html=True)