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
    page_title="Enhanced Butane Tank Thermal Simulation", 
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧪 Butane Tank Thermal Simulation</h1>
    <p style="text-align: center; color: white; margin: 0;">
        Fully insulated system with tank materials and heater modeling
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
    """Calculate enthalpy of vaporization as function of temperature"""
    T_ref = 272.05
    if T >= T_critical:
        return 0
    ratio = (T_critical - T) / (T_critical - T_ref)
    return max(h_vap_normal_kg * (ratio ** 0.38), 0)

def simulate_butane_tank_enhanced(initial_temp_c, initial_mass, target_flow_rate,
                                dt, max_time, tank_mass, tank_material, 
                                heater_power, tank_initial_temp_c):
    """Enhanced simulation with tank materials, full insulation, and heater"""
    
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
        'heater_status': [1.0 if heater_power > 0 else 0.0]
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
        
        # Calculate enthalpy of vaporization at current temperature
        h_vap_current = calculate_enthalpy_vaporization(T_liquid)
        
        # Heat removed by vaporization (cooling effect)
        Q_removed_vaporization = m_dot_actual * h_vap_current  # W (J/s)
        
        # Heat added by heater (heating effect)
        Q_added_heater = heater_power  # W
        
        # Net heat transfer
        Q_net = Q_added_heater - Q_removed_vaporization  # W
        
        # Combined thermal mass (liquid + tank)
        # Assuming thermal equilibrium between liquid and tank
        C_liquid = m_liquid * c_p_liquid  # J/K
        C_tank = tank_mass * tank_heat_capacity  # J/K
        C_total = C_liquid + C_tank  # J/K
        
        # Temperature change (fully insulated system - no ambient heat transfer)
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
        results['temperature_combined'].append(T_liquid - 273.15)  # Same due to equilibrium
        results['pressure'].append(P_vapor / 1e5)
        results['mass'].append(m_liquid)
        results['flow_rate'].append(m_dot_actual * 1e6)  # Convert to μg/s
        results['h_vap'].append(h_vap_current / 1000)  # Convert to kJ/kg
        results['heat_removed_vaporization'].append(Q_removed_vaporization)
        results['heat_added_heater'].append(Q_added_heater)
        results['heat_net'].append(Q_net)
        results['heater_status'].append(1.0 if heater_power > 0 else 0.0)
        
        # Safety checks
        if T_liquid < T_triple or T_liquid > T_critical:
            break
    
    return {k: np.array(v) for k, v in results.items()}

def create_interactive_plots(results):
    """Create enhanced interactive Plotly plots"""
    
    # Create subplots with 2x3 layout
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Temperature Evolution', 'Pressure Evolution', 
                       'Mass Depletion', 'Flow Rate',
                       'Heat Transfer Analysis', 'Heater Performance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    time_hours = results['time'] / 3600
    
    # Temperature plot
    fig.add_trace(
        go.Scatter(x=time_hours, y=results['temperature_combined'], 
                  name='System Temperature', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Pressure plot
    fig.add_trace(
        go.Scatter(x=time_hours, y=results['pressure'], 
                  name='Pressure', line=dict(color='green', width=2)),
        row=1, col=2
    )
    
    # Mass plot
    fig.add_trace(
        go.Scatter(x=time_hours, y=results['mass'], 
                  name='Mass', line=dict(color='red', width=2)),
        row=2, col=1
    )
    
    # Flow rate plot
    fig.add_trace(
        go.Scatter(x=time_hours, y=results['flow_rate'], 
                  name='Flow Rate', line=dict(color='purple', width=2)),
        row=2, col=2
    )
    
    # Heat transfer analysis
    fig.add_trace(
        go.Scatter(x=time_hours, y=results['heat_added_heater'], 
                  name='Heat Added (Heater)', line=dict(color='orange')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_hours, y=-results['heat_removed_vaporization'], 
                  name='Heat Removed (Vaporization)', line=dict(color='cyan')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_hours, y=results['heat_net'], 
                  name='Net Heat Transfer', line=dict(color='black', width=3)),
        row=3, col=1
    )
    
    # Heater performance
    fig.add_trace(
        go.Scatter(x=time_hours, y=results['heater_status'] * 100, 
                  name='Heater Status (%)', line=dict(color='red', width=2)),
        row=3, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=3, col=2)
    
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure (bar)", row=1, col=2)
    fig.update_yaxes(title_text="Mass (kg)", row=2, col=1)
    fig.update_yaxes(title_text="Flow Rate (μg/s)", row=2, col=2)
    fig.update_yaxes(title_text="Heat Transfer Rate (W)", row=3, col=1)
    fig.update_yaxes(title_text="Heater Status (%)", row=3, col=2)
    
    fig.update_layout(height=800, showlegend=False)
    
    return fig

def validate_inputs(params):
    """Validate user inputs"""
    errors = []
    warnings = []
    
    if params['initial_temp'] < -100 or params['initial_temp'] > 100:
        errors.append("Initial temperature should be between -100°C and 100°C")
    
    if params['initial_mass'] <= 0:
        errors.append("Initial mass must be positive")
    
    if params['flow_rate'] <= 0:
        errors.append("Flow rate must be positive")
    
    if params['time_step'] <= 0 or params['time_step'] > 10:
        errors.append("Time step should be between 0 and 10 seconds")
    
    if params['tank_mass'] <= 0:
        errors.append("Tank mass must be positive")
    
    if params['heater_power'] < 0 or params['heater_power'] > 100:
        errors.append("Heater power should be between 0 and 100 W")
    
    if params['flow_rate'] > 1e-3:
        warnings.append("High flow rate may cause rapid depletion")
    
    if params['heater_power'] < 1 and params['flow_rate'] > 1e-6:
        warnings.append("Low heater power may not compensate for cooling from vaporization")
    
    return errors, warnings

# Sidebar for parameters
with st.sidebar:
    st.header("🔧 Simulation Parameters")
    
    # Basic parameters
    st.subheader("System Configuration")
    params = {}
    params['initial_temp'] = st.number_input("Initial Liquid Temperature (°C)", value=20.0, step=1.0)
    params['flow_rate'] = st.number_input("Target Flow Rate (kg/s)", value=1e-5, format="%.1e", step=1e-6)
    params['initial_mass'] = st.number_input("Initial Butane Mass (kg)", value=0.225, step=0.001, min_value=0.001, format="%.3f")
    
    # Tank configuration
    st.subheader("🏗️ Tank Configuration")
    params['tank_mass'] = st.number_input("Tank Mass (kg)", value=1.0, step=0.1, min_value=0.1)
    params['tank_material'] = st.selectbox("Tank Material", options=list(TANK_MATERIALS.keys()), index=0)
    params['tank_initial_temp'] = st.number_input("Tank Initial Temperature (°C)", value=20.0, step=1.0)
    
    # Display selected material properties
    selected_material = TANK_MATERIALS[params['tank_material']]
    st.info(f"""
    **{params['tank_material']} Properties:**
    - Heat Capacity: {selected_material['heat_capacity']} J/(kg·K)
    - Density: {selected_material['density']} kg/m³
    - Thermal Conductivity: {selected_material['thermal_conductivity']} W/(m·K)
    """)
    
    # Heater configuration
    st.subheader("🔥 Heater Configuration")
    params['heater_power'] = st.number_input("Heater Power (W)", value=5.0, step=0.5, min_value=0.0, max_value=100.0)
    
    if params['heater_power'] > 0:
        st.success(f"✅ Heater enabled: {params['heater_power']} W")
    else:
        st.warning("⚠️ Heater disabled")
    
    # Simulation settings
    st.subheader("⏱️ Simulation Settings")
    params['max_time'] = st.number_input("Max Simulation Time (s)", value=50000, step=1000, min_value=1000)
    params['time_step'] = st.number_input("Time Step (s)", value=1.0, step=0.1, min_value=0.1)
    
    # Insulation notice
    st.markdown("""
    <div class="insulated-box">
    <h4>🛡️ Fully Insulated System</h4>
    <p>This simulation models a fully insulated tank with:</p>
    <ul>
    <li>❄️ No external heat transfer</li>
    <li>🔥 Internal heater compensation</li>
    <li>🎯 Pure thermal balance between heater and vaporization</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Validation
    errors, warnings = validate_inputs(params)
    
    if errors:
        st.error("❌ Input Errors:")
        for error in errors:
            st.error(f"• {error}")
    
    if warnings:
        st.warning("⚠️ Warnings:")
        for warning in warnings:
            st.warning(f"• {warning}")
    
    # Run simulation button
    run_simulation = st.button("🚀 Run Simulation", disabled=bool(errors))
    
    # Advanced options
    with st.expander("📊 Display Options"):
        show_data_table = st.checkbox("Show Data Table", value=False)
        export_data = st.checkbox("Enable Data Export", value=False)

# Main content area
if run_simulation:
    with st.spinner("🔄 Running enhanced thermal simulation..."):
        results = simulate_butane_tank_enhanced(
            initial_temp_c=params['initial_temp'],
            initial_mass=params['initial_mass'],
            target_flow_rate=params['flow_rate'],
            dt=params['time_step'],
            max_time=params['max_time'],
            tank_mass=params['tank_mass'],
            tank_material=params['tank_material'],
            heater_power=params['heater_power'],
            tank_initial_temp_c=params['tank_initial_temp']
        )
    
    if len(results['time']) > 0:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp_change = results['temperature_combined'][-1] - params['initial_temp']
            st.metric("Final Temperature", f"{results['temperature_combined'][-1]:.1f}°C", 
                     f"{temp_change:+.1f}°C")
        
        with col2:
            pressure_change = results['pressure'][-1] - results['pressure'][0]
            st.metric("Final Pressure", f"{results['pressure'][-1]:.2f} bar",
                     f"{pressure_change:+.2f} bar")
        
        with col3:
            mass_change = results['mass'][-1] - params['initial_mass']
            st.metric("Mass Remaining", f"{results['mass'][-1]:.3f} kg",
                     f"{mass_change:.3f} kg")
        
        with col4:
            st.metric("Simulation Time", f"{results['time'][-1]/3600:.1f} hours")
        
        # Energy balance summary
        st.subheader("⚡ Energy Balance Summary")
        col1, col2, col3 = st.columns(3)
        
        total_heat_added = np.sum(results['heat_added_heater']) * params['time_step']
        total_heat_removed = np.sum(results['heat_removed_vaporization']) * params['time_step']
        net_energy = total_heat_added - total_heat_removed
        
        with col1:
            st.metric("Total Heat Added", f"{total_heat_added/1000:.1f} kJ", "Heater")
        
        with col2:
            st.metric("Total Heat Removed", f"{total_heat_removed/1000:.1f} kJ", "Vaporization")
        
        with col3:
            st.metric("Net Energy Balance", f"{net_energy/1000:+.1f} kJ")
        
        # Main plots
        st.subheader("📊 Simulation Results")
        fig_main = create_interactive_plots(results)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # System analysis
        st.subheader("🔍 System Analysis")
        
        # Calculate efficiency metrics
        avg_heater_power = np.mean(results['heat_added_heater'])
        avg_cooling_power = np.mean(results['heat_removed_vaporization'])
        heater_efficiency = (avg_heater_power / avg_cooling_power * 100) if avg_cooling_power > 0 else 0
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.info(f"""
            **Thermal Performance:**
            - Tank Material: {params['tank_material']}
            - Tank Heat Capacity: {TANK_MATERIALS[params['tank_material']]['heat_capacity']} J/(kg·K)
            - Average Heater Power: {avg_heater_power:.2f} W
            - Average Cooling Power: {avg_cooling_power:.2f} W
            """)
        
        with analysis_col2:
            if heater_efficiency < 80:
                efficiency_color = "🔴"
            elif heater_efficiency < 100:
                efficiency_color = "🟡"
            else:
                efficiency_color = "🟢"
            
            st.info(f"""
            **Heater Performance:**
            - Heater Efficiency: {efficiency_color} {heater_efficiency:.1f}%
            - Temperature Stability: {'✅ Good' if abs(temp_change) < 5 else '⚠️ Poor'}
            - System Status: {'🔥 Heating' if net_energy > 0 else '❄️ Cooling'}
            """)
        
        # Data table
        if show_data_table:
            st.subheader("📋 Simulation Data")
            df = pd.DataFrame({
                'Time (h)': results['time'] / 3600,
                'Temperature (°C)': results['temperature_combined'],
                'Pressure (bar)': results['pressure'],
                'Mass (kg)': results['mass'],
                'Flow Rate (μg/s)': results['flow_rate'],
                'H_vap (kJ/kg)': results['h_vap'],
                'Heater Power (W)': results['heat_added_heater'],
                'Cooling Power (W)': results['heat_removed_vaporization'],
                'Net Heat (W)': results['heat_net']
            })
            st.dataframe(df, use_container_width=True)
        
        # Export functionality
        if export_data:
            st.subheader("💾 Export Data")
            
            # Create CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="butane_enhanced_simulation_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create detailed report
                report = f"""
# Enhanced Butane Tank Simulation Report

## System Configuration
- Initial Temperature: {params['initial_temp']}°C
- Initial Mass: {params['initial_mass']} kg
- Target Flow Rate: {params['flow_rate']:.2e} kg/s
- Tank Material: {params['tank_material']}
- Tank Mass: {params['tank_mass']} kg
- Heater Power: {params['heater_power']} W

## Material Properties
- Heat Capacity: {TANK_MATERIALS[params['tank_material']]['heat_capacity']} J/(kg·K)
- Density: {TANK_MATERIALS[params['tank_material']]['density']} kg/m³

## Simulation Results
- Final Temperature: {results['temperature_combined'][-1]:.1f}°C
- Temperature Change: {temp_change:+.1f}°C
- Final Pressure: {results['pressure'][-1]:.2f} bar
- Mass Remaining: {results['mass'][-1]:.3f} kg
- Simulation Duration: {results['time'][-1]/3600:.1f} hours

## Energy Balance
- Total Heat Added: {total_heat_added/1000:.1f} kJ
- Total Heat Removed: {total_heat_removed/1000:.1f} kJ
- Net Energy: {net_energy/1000:+.1f} kJ
- Heater Efficiency: {heater_efficiency:.1f}%

## System Status
- Fully Insulated: ✅ Yes
- External Heat Transfer: ❌ None
- Heater Compensation: {'✅ Active' if params['heater_power'] > 0 else '❌ Disabled'}
                """
                
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name="butane_enhanced_simulation_report.md",
                    mime="text/markdown"
                )
        
        st.success("✅ Enhanced simulation completed successfully!")
        
    else:
        st.error("❌ Simulation failed. Please check your parameters.")

else:
    # Welcome message and instructions
    st.markdown("""
    ## Welcome to the Enhanced Butane Tank Thermal Simulation! 🎯
    
    This advanced simulation now includes:
    
    ### 🆕 New Features:
    - **🏗️ Tank Materials:** Choose from aluminum, stainless steel variants, and titanium
    - **🛡️ Full Insulation:** No external heat transfer - pure internal thermal balance
    - **🔥 Heater Model:** Built-in 5W heater with configurable power to compensate energy loss
    - **⚡ Energy Balance:** Detailed tracking of heating vs. cooling effects
    
    ### 🔬 Tank Materials Available:
    """)
    
    # Display material comparison table
    materials_df = pd.DataFrame.from_dict(TANK_MATERIALS, orient='index')
    materials_df.index.name = 'Material'
    materials_df.columns = ['Heat Capacity (J/kg·K)', 'Density (kg/m³)', 'Thermal Conductivity (W/m·K)']
    st.dataframe(materials_df, use_container_width=True)
    
    st.markdown("""
    ### 🔥 Heater System:
    - **Default Power:** 5W (configurable 0-100W)
    - **Purpose:** Compensate for cooling from butane vaporization
    - **Control:** Constant power output (future versions may include temperature control)
    - **Efficiency:** System tracks heater performance vs. cooling demand
    
    ### 🛡️ Fully Insulated System:
    - No ambient heat transfer
    - Pure balance between internal heating and cooling
    - Realistic modeling of isolated tank behavior
    - Focus on internal thermal dynamics
    
    ### How to Use:
    1. 🔧 Select tank material and configure mass
    2. 🔥 Set heater power (default 5W)
    3. 🚀 Run simulation to see thermal balance
    4. 📊 Analyze heater efficiency and temperature stability
    
    **Ready to explore the enhanced thermal dynamics?** Configure your system and run the simulation! 🚀
    """)
    
    # Display key equations and theory
    with st.expander("📚 Enhanced Physical Model"):
        st.markdown("""
        ### Energy Balance Equation:
        ```
        Q_net = Q_heater - Q_vaporization
        dT/dt = Q_net / (m_liquid × C_p_liquid + m_tank × C_p_tank)
        ```
        
        ### Tank Materials Impact:
        - **Heat Capacity:** Higher values provide more thermal mass (temperature stability)
        - **Aluminum 6061:** Lightweight, high heat capacity (896 J/kg·K)
        - **Stainless Steel:** Moderate heat capacity (500 J/kg·K), corrosion resistant
        - **Titanium TC-4:** High strength, moderate heat capacity (523 J/kg·K)
        
        ### Heater Compensation:
        - **Cooling Load:** Q_vaporization = ṁ × h_vap(T)
        - **Heating Input:** Q_heater = P_heater (constant)
        - **Net Effect:** Temperature rises if Q_heater > Q_vaporization
        
        ### Fully Insulated Assumption:
        - No conduction, convection, or radiation losses
        - All energy changes from internal processes only
        - Realistic for well-insulated laboratory/industrial setups
        """)
    
    # Add material selection guide
    st.markdown("""
    ### 🎯 Material Selection Guide:
    """)
    
    guide_col1, guide_col2 = st.columns(2)
    
    with guide_col1:
        st.info("""
        **For Temperature Stability:**
        - Choose **Aluminum 6061** (highest heat capacity)
        - Higher tank mass for more thermal inertia
        - Match heater power to expected cooling load
        """)
    
    with guide_col2:
        st.info("""
        **For Realistic Systems:**
        - **Stainless Steel 316L** for chemical compatibility
        - Consider actual tank wall thickness
        - 5W heater typical for small research systems
        """)