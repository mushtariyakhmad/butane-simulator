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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧪 Advanced Butane Tank Thermal Simulation</h1>
    <p style="text-align: center; color: white; margin: 0;">
        Real-time thermal analysis with NIST properties and interactive visualization
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

@st.cache_data
def simulate_butane_tank_nist(initial_temp_c, initial_mass, target_flow_rate,
                               dt, max_time, ambient_temp_c, heat_transfer_coeff):
    """Enhanced simulation with detailed tracking"""
    
    T_kelvin = initial_temp_c + 273.15
    T_ambient = ambient_temp_c + 273.15
    m_liquid = initial_mass
    R_specific = R / M_butane
    P_initial = antoine_vapor_pressure(T_kelvin)
    A_orifice = calculate_orifice_area(target_flow_rate, P_initial, R_specific, T_kelvin, gamma)

    # Initialize arrays
    results = {
        'time': [],
        'temperature': [],
        'pressure': [],
        'mass': [],
        'flow_rate': [],
        'h_vap': [],
        'heat_removed': [],
        'heat_ambient': [],
        'heat_net': []
    }

    t = 0
    while (t < max_time and m_liquid > 0.001 and T_kelvin > T_triple and T_kelvin < T_critical):
        P_vapor = antoine_vapor_pressure(T_kelvin)
        if P_vapor <= 0:
            break

        m_dot_actual = calculate_mass_flow_from_area(A_orifice, P_vapor, R_specific, T_kelvin, gamma)
        m_dot_actual = min(m_dot_actual, m_liquid / dt)
        h_vap_current = calculate_enthalpy_vaporization(T_kelvin)

        # Energy balance
        Q_removed = m_dot_actual * h_vap_current * dt
        Q_ambient = heat_transfer_coeff * (T_ambient - T_kelvin) * dt
        Q_net = Q_ambient - Q_removed

        # Update state
        dT = Q_net / (m_liquid * c_p_liquid) if m_liquid > 0 else 0
        T_kelvin += dT
        m_liquid -= m_dot_actual * dt
        t += dt

        # Store results
        results['time'].append(t)
        results['temperature'].append(T_kelvin - 273.15)
        results['pressure'].append(P_vapor / 1e5)
        results['mass'].append(m_liquid)
        results['flow_rate'].append(m_dot_actual * 1e6)
        results['h_vap'].append(h_vap_current / 1000)
        results['heat_removed'].append(Q_removed)
        results['heat_ambient'].append(Q_ambient)
        results['heat_net'].append(Q_net)

    return {k: np.array(v) for k, v in results.items()}

def create_interactive_plots(results):
    """Create interactive Plotly plots"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature Evolution', 'Pressure Evolution', 
                       'Mass Depletion', 'Flow Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    time_hours = results['time'] / 3600
    
    # Temperature plot
    fig.add_trace(
        go.Scatter(x=time_hours, y=results['temperature'], 
                  name='Temperature', line=dict(color='blue', width=2)),
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
    
    # Update layout
    fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
    
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure (bar)", row=1, col=2)
    fig.update_yaxes(title_text="Mass (kg)", row=2, col=1)
    fig.update_yaxes(title_text="Flow Rate (μg/s)", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    
    return fig

def create_energy_balance_plot(results):
    """Create energy balance visualization"""
    fig = go.Figure()
    
    time_hours = results['time'] / 3600
    
    fig.add_trace(go.Scatter(
        x=time_hours, y=results['heat_ambient'],
        name='Heat from Ambient', line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=time_hours, y=-results['heat_removed'],
        name='Heat Removed (Vaporization)', line=dict(color='cyan')
    ))
    
    fig.add_trace(go.Scatter(
        x=time_hours, y=results['heat_net'],
        name='Net Heat Transfer', line=dict(color='black', width=3)
    ))
    
    fig.update_layout(
        title='Energy Balance Analysis',
        xaxis_title='Time (hours)',
        yaxis_title='Heat Transfer Rate (W)',
        height=400
    )
    
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
    
    if params['initial_temp'] > params['ambient_temp'] + 50:
        warnings.append("Large temperature difference may cause rapid cooling")
    
    if params['flow_rate'] > 1e-3:
        warnings.append("High flow rate may cause rapid depletion")
    
    return errors, warnings

# Sidebar for parameters
with st.sidebar:
    st.header("🔧 Simulation Parameters")
    
    # Input parameters
    params = {}
    params['initial_temp'] = st.number_input("Initial Temperature (°C)", value=20.0, step=1.0)
    params['flow_rate'] = st.number_input("Target Flow Rate (kg/s)", value=1e-5, format="%.1e", step=1e-6)
    params['initial_mass'] = st.number_input( "Initial Mass (kg)",value=0.225, step=0.001, min_value=0.001,format="%.3f")
    params['ambient_temp'] = st.number_input("Ambient Temperature (°C)", value=20.0, step=1.0)
    params['max_time'] = st.number_input("Max Simulation Time (s)", value=50000, step=1000, min_value=1000)
    params['time_step'] = st.number_input("Time Step (s)", value=1.0, step=0.1, min_value=0.1)
    params['heat_coeff'] = st.number_input("Heat Transfer Coefficient (W/K)", value=0.1, step=0.01, min_value=0.01)
    
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
    with st.expander("Advanced Options"):
        show_energy_balance = st.checkbox("Show Energy Balance", value=True)
        show_data_table = st.checkbox("Show Data Table", value=False)
        export_data = st.checkbox("Enable Data Export", value=False)

# Main content area
if run_simulation:
    with st.spinner("🔄 Running thermal simulation..."):
        results = simulate_butane_tank_nist(
            initial_temp_c=params['initial_temp'],
            initial_mass=params['initial_mass'],
            target_flow_rate=params['flow_rate'],
            dt=params['time_step'],
            max_time=params['max_time'],
            ambient_temp_c=params['ambient_temp'],
            heat_transfer_coeff=params['heat_coeff']
        )
    
    if len(results['time']) > 0:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Temperature", f"{results['temperature'][-1]:.1f}°C", 
                     f"{results['temperature'][-1] - params['initial_temp']:.1f}°C")
        
        with col2:
            st.metric("Final Pressure", f"{results['pressure'][-1]:.2f} bar",
                     f"{results['pressure'][-1] - results['pressure'][0]:.2f} bar")
        
        with col3:
            st.metric("Mass Remaining", f"{results['mass'][-1]:.3f} kg",
                     f"{results['mass'][-1] - params['initial_mass']:.3f} kg")
        
        with col4:
            st.metric("Simulation Time", f"{results['time'][-1]/3600:.1f} hours")
        
        # Main plots
        st.subheader("📊 Simulation Results")
        fig_main = create_interactive_plots(results)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Energy balance plot
        if show_energy_balance:
            st.subheader("⚡ Energy Balance Analysis")
            fig_energy = create_energy_balance_plot(results)
            st.plotly_chart(fig_energy, use_container_width=True)
        
        # Data table
        if show_data_table:
            st.subheader("📋 Simulation Data")
            df = pd.DataFrame({
                'Time (h)': results['time'] / 3600,
                'Temperature (°C)': results['temperature'],
                'Pressure (bar)': results['pressure'],
                'Mass (kg)': results['mass'],
                'Flow Rate (μg/s)': results['flow_rate'],
                'H_vap (kJ/kg)': results['h_vap']
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
                    file_name="butane_simulation_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create summary report
                report = f"""
# Butane Tank Simulation Report

## Parameters
- Initial Temperature: {params['initial_temp']}°C
- Initial Mass: {params['initial_mass']} kg
- Target Flow Rate: {params['flow_rate']:.2e} kg/s
- Ambient Temperature: {params['ambient_temp']}°C

## Results
- Final Temperature: {results['temperature'][-1]:.1f}°C
- Final Pressure: {results['pressure'][-1]:.2f} bar
- Mass Remaining: {results['mass'][-1]:.3f} kg
- Simulation Duration: {results['time'][-1]/3600:.1f} hours
                """
                
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name="butane_simulation_report.md",
                    mime="text/markdown"
                )
        
        st.success("✅ Simulation completed successfully!")
        
    else:
        st.error("❌ Simulation failed. Please check your parameters.")

else:
    # Welcome message and instructions
    st.markdown("""
    ## Welcome to the Butane Tank Thermal Simulation! 🎯
    
    This advanced simulation tool helps you analyze the thermal behavior of butane tanks during gas flow operations.
    
    ### Features:
    - **Real-time thermal modeling** with NIST-validated butane properties
    - **Interactive visualizations** with Plotly charts
    - **Energy balance analysis** showing heat transfer dynamics
    - **Data export capabilities** for further analysis
    - **Input validation** with warnings and error checking
    
    ### How to Use:
    1. 🔧 Adjust simulation parameters in the sidebar
    2. 🚀 Click "Run Simulation" to start
    3. 📊 Analyze results with interactive plots
    4. 💾 Export data if needed
    
    ### Physical Model:
    The simulation uses the **Antoine equation** for vapor pressure calculation and models:
    - Choked flow through orifices
    - Temperature-dependent enthalpy of vaporization
    - Heat transfer from ambient environment
    - Real-time mass balance
    
    **Ready to start?** Configure your parameters and run the simulation! 🚀
    """)
    
    # Display physical constants
    with st.expander("📚 Physical Constants & Properties"):
        const_col1, const_col2 = st.columns(2)
        
        with const_col1:
            st.markdown("""
            **Butane Properties:**
            - Molecular Weight: 58.12 g/mol
            - Critical Temperature: 152.0°C
            - Critical Pressure: 37.96 bar
            - Boiling Point: -0.5°C
            """)
        
        with const_col2:
            st.markdown("""
            **Simulation Constants:**
            - Gas Constant: 8.314 J/(mol·K)
            - Heat Capacity (liquid): 2.4 kJ/(kg·K)
            - Heat Ratio (γ): 1.10
            - Triple Point: -138.3°C
            """)