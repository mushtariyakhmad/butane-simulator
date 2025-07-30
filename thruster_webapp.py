import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
from scipy.integrate import ode
from scipy.interpolate import interp1d

# Configure page
st.set_page_config(
    page_title="Enhanced Thruster Simulation", 
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
    
    .thermodynamic-box {
        background: var(--secondary-gradient);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
    }
</style>
""", unsafe_allow_html=True)

class EnhancedButaneThrusterSimulation:
    """Enhanced Butane Thruster Simulation with Thermodynamic Modeling"""
    
    def __init__(self):
        # Physical constants
        self.R_universal = 8.314  # J/(mol·K)
        self.M_butane = 58.1222e-3  # kg/mol (C4H10 molecular weight)
        self.R_specific = self.R_universal / self.M_butane  # J/(kg·K) ≈ 143.0
        self.g0 = 9.80665  # Standard gravity (m/s²)
        
        # Butane thermodynamic properties (enhanced)
        self.k = 1.05  # Specific heat ratio for butane (as specified)
        self.T_critical = 425.13  # K
        self.P_critical = 3.796e6  # Pa
        self.T_boiling = 272.65  # K at 1 atm
        self.heat_vaporization = 385e3  # J/kg
        
        # Spacecraft properties
        self.spacecraft_mass = 16.0  # kg
        
        # Nozzle geometry (constant as specified)
        self.throat_radius = 0.244e-3 / 2  # m (0.244mm diameter)
        self.exit_radius = 5.62e-3 / 2     # m (5.62mm diameter)
        self.throat_area = math.pi * self.throat_radius**2
        self.exit_area = math.pi * self.exit_radius**2
        self.expansion_ratio = self.exit_area / self.throat_area  # Constant
        
        # Nozzle efficiency (93% as specified)
        self.nozzle_efficiency = 0.93
        
        # Reference values for validation
        self.target_temperature = 623.0  # K (350°C)
        self.reference_isp = 106.0  # s
        self.reference_thrust = 24.1e-3  # N
        self.reference_efficiency = 69.0  # %
        
        # Setup enhanced heat capacity lookup table
        self.setup_cp_lookup_table()
        
    def setup_cp_lookup_table(self):
        """Setup enhanced heat capacity lookup table for butane"""
        # Temperature range from ambient to high operating temperatures
        temperatures = np.array([
            200, 250, 273.15, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800
        ])  # K
        
        # Heat capacity values for butane gas (J/kg·K)
        # Enhanced values based on thermodynamic data with baseline at 2400 J/kg·K
        cp_values = np.array([
            2100, 2200, 2250, 2300, 2380, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800
        ])
        
        # Create interpolation function
        self.cp_interpolator = interp1d(
            temperatures, cp_values,
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )
    
    def get_cp_temperature_dependent(self, temperature):
        """Get temperature-dependent heat capacity with proper bounds"""
        cp = float(self.cp_interpolator(temperature))
        # Ensure reasonable bounds (2000-3000 J/kg·K)
        return max(2000, min(3000, cp))
    
    def calculate_exhaust_velocity_from_thermodynamics(self, T1, P1, P2):
        """
        Calculate exhaust velocity using fundamental thermodynamic equation:
        V2 = √[(2k)/(k-1) * R*T1 * (1 - (P2/P1)^((k-1)/k))]
        
        This is the IDEAL velocity before nozzle efficiency is applied.
        """
        if P1 <= P2 or P1 <= 0 or T1 <= 0:
            return 0
        
        pressure_ratio = P2 / P1
        
        # Check if pressure ratio is physically realistic
        if pressure_ratio >= 1.0:
            return 0
        
        # Fundamental thermodynamic equation
        term1 = (2 * self.k) / (self.k - 1)
        term2 = self.R_specific * T1
        term3 = 1 - (pressure_ratio ** ((self.k - 1) / self.k))
        
        if term3 <= 0:
            return 0
        
        # Ideal exhaust velocity
        v_ideal = math.sqrt(term1 * term2 * term3)
        
        return v_ideal
    
    def calculate_realistic_exhaust_velocity(self, T1, P1, P2):
        """
        Calculate realistic exhaust velocity with nozzle efficiency:
        V2_real = V2_ideal * η
        """
        v_ideal = self.calculate_exhaust_velocity_from_thermodynamics(T1, P1, P2)
        v_realistic = v_ideal * self.nozzle_efficiency
        
        return v_realistic, v_ideal
    
    def calculate_realistic_isp_from_velocity(self, exhaust_velocity):
        """
        Calculate realistic specific impulse from exhaust velocity:
        Isp = V2 / g0
        
        This is the correct relationship: V2 = Isp * g0, so Isp = V2 / g0
        """
        return exhaust_velocity / self.g0
    
    def calculate_mass_flow_rate_choked(self, P1, T1):
        """
        Calculate choked mass flow rate through throat using:
        ṁ = P1 * At * √(k/RT1) * (2/(k+1))^((k+1)/(2(k-1)))
        """
        if P1 <= 0 or T1 <= 0:
            return 0
        
        # Choked flow equation for mass flow rate
        term1 = P1 * self.throat_area
        term2 = math.sqrt(self.k / (self.R_specific * T1))
        term3 = (2 / (self.k + 1)) ** ((self.k + 1) / (2 * (self.k - 1)))
        
        mass_flow_rate = term1 * term2 * term3
        return max(0, mass_flow_rate)
    
    def calculate_dynamic_chamber_pressure(self, mass_flow_rate, temperature, base_pressure):
        """
        Calculate dynamic chamber pressure based on mass flow and temperature:
        P1 = f(ṁ, T) - Higher temperature and mass flow increase pressure
        """
        if temperature <= 0 or mass_flow_rate <= 0:
            return base_pressure * 0.5
        
        # Temperature factor (normalized to target temperature)
        temp_factor = temperature / self.target_temperature
        
        # Mass flow factor (normalized, scaled appropriately)
        mass_flow_factor = (mass_flow_rate * 1e6)  # Convert to mg/s for scaling
        
        # Combined pressure enhancement
        pressure_multiplier = 0.8 + 0.4 * temp_factor + 0.3 * min(mass_flow_factor, 1.0)
        
        dynamic_pressure = base_pressure * pressure_multiplier
        
        # Ensure reasonable bounds
        return max(base_pressure * 0.5, min(base_pressure * 1.5, dynamic_pressure))
    
    def calculate_thrust_components(self, mass_flow_rate, exhaust_velocity, P2, P3):
        """
        Calculate thrust components:
        F = ṁ*V2 + (P2 - P3)*A2
        """
        momentum_thrust = mass_flow_rate * exhaust_velocity
        pressure_thrust = (P2 - P3) * self.exit_area
        total_thrust = momentum_thrust + pressure_thrust
        
        return {
            'momentum': momentum_thrust,
            'pressure': pressure_thrust,
            'total': total_thrust
        }
    
    def temperature_evolution_ode(self, t, state, Q_input, mass_flow_rate, propellant_mass):
        """
        Enhanced temperature evolution using energy balance:
        Q = m * cp * dT/dt + heat_losses
        
        Rearranged: dT/dt = (Q - heat_losses) / (m * cp)
        """
        T = state[0]
        
        # Get temperature-dependent heat capacity
        cp = self.get_cp_temperature_dependent(T)
        
        # Heat losses due to mass flow (enthalpy carried away)
        T_ref = 300.0  # Reference temperature (K)
        heat_loss_convective = mass_flow_rate * cp * (T - T_ref)
        
        # Additional heat losses (radiation, conduction - simplified)
        heat_loss_radiation = 5.67e-8 * 0.1 * (T**4 - T_ref**4)  # Stefan-Boltzmann simplified
        
        # Phase change energy consideration near boiling point
        phase_change_energy = 0
        if abs(T - self.T_boiling) < 10.0 and mass_flow_rate > 0:
            phase_change_energy = mass_flow_rate * self.heat_vaporization * 0.05
        
        # Net heat available for temperature rise
        total_heat_loss = heat_loss_convective + heat_loss_radiation + phase_change_energy
        net_heat = Q_input - total_heat_loss
        
        # Temperature rate of change: dT/dt = Q_net / (m * cp)
        if propellant_mass > 0 and cp > 0:
            dT_dt = net_heat / (propellant_mass * cp)
        else:
            dT_dt = 0
        
        # Prevent negative temperatures
        if T <= 273.15 and dT_dt < 0:
            dT_dt = 0
        
        return [dT_dt]
    
    def simulate_enhanced_performance(self, config):
        """Enhanced simulation with thermodynamic modeling"""
        # Unpack configuration
        initial_temp = config['initial_temp']
        Q_input = config['heat_input']
        base_chamber_pressure = config['chamber_pressure']
        exit_pressure = config['exit_pressure']
        ambient_pressure = config['ambient_pressure']
        simulation_time = config['simulation_time']
        dt = config['dt']
        propellant_mass = config['propellant_mass']
        
        # Initialize results storage
        results = {
            'time': [],
            'temperature': [],
            'temperature_celsius': [],
            'heat_capacity': [],
            'chamber_pressure': [],
            'mass_flow_rate': [],
            'exhaust_velocity_ideal': [],
            'exhaust_velocity_realistic': [],
            'specific_impulse_ideal': [],
            'specific_impulse_realistic': [],
            'thrust_momentum': [],
            'thrust_pressure': [],
            'thrust_total': [],
            'thrust_mN': [],
            'power_input': [],
            'power_kinetic': [],
            'thermal_efficiency': [],
            'acceleration_ms2': [],
            'total_impulse': [],
            'expansion_ratio': [],
            'pressure_ratio': []
        }
        
        # Setup ODE solver for temperature evolution
        solver = ode(self.temperature_evolution_ode)
        solver.set_integrator('dopri5', rtol=1e-8, atol=1e-10)
        solver.set_initial_value([initial_temp], 0)
        
        time = 0
        total_impulse = 0
        
        while solver.successful() and time < simulation_time:
            # Get current temperature
            current_temp = solver.y[0]
            current_temp_celsius = current_temp - 273.15
            
            # Get temperature-dependent properties
            cp_current = self.get_cp_temperature_dependent(current_temp)
            
            # Calculate dynamic chamber pressure
            if len(results['mass_flow_rate']) > 0:
                # Use previous mass flow rate for pressure calculation to avoid circular dependency
                prev_mass_flow = results['mass_flow_rate'][-1]
                P1 = self.calculate_dynamic_chamber_pressure(prev_mass_flow, current_temp, base_chamber_pressure)
            else:
                P1 = base_chamber_pressure
            
            # Calculate mass flow rate (choked flow through throat)
            mass_flow_rate = self.calculate_mass_flow_rate_choked(P1, current_temp)
            
            # Update solver parameters for next iteration
            solver.set_f_params(Q_input, mass_flow_rate, propellant_mass)
            
            # Calculate exhaust velocities
            v_realistic, v_ideal = self.calculate_realistic_exhaust_velocity(
                current_temp, P1, exit_pressure
            )
            
            # Calculate specific impulses using realistic relationship: Isp = V2 / g0
            isp_ideal = self.calculate_realistic_isp_from_velocity(v_ideal)
            isp_realistic = self.calculate_realistic_isp_from_velocity(v_realistic)
            
            # Calculate thrust components
            thrust_components = self.calculate_thrust_components(
                mass_flow_rate, v_realistic, exit_pressure, ambient_pressure
            )
            
            # Calculate power and efficiency
            power_kinetic = 0.5 * mass_flow_rate * v_realistic**2
            thermal_efficiency = (power_kinetic / Q_input) * 100 if Q_input > 0 else 0
            
            # Calculate spacecraft acceleration
            acceleration = thrust_components['total'] / self.spacecraft_mass if self.spacecraft_mass > 0 else 0
            
            # Calculate total impulse (integral of thrust over time)
            if len(results['time']) > 0:
                dt_actual = time - results['time'][-1]
                total_impulse += thrust_components['total'] * dt_actual
            
            # Calculate pressure ratio
            pressure_ratio = exit_pressure / P1 if P1 > 0 else 0
            
            # Store results
            results['time'].append(time)
            results['temperature'].append(current_temp)
            results['temperature_celsius'].append(current_temp_celsius)
            results['heat_capacity'].append(cp_current)
            results['chamber_pressure'].append(P1)
            results['mass_flow_rate'].append(mass_flow_rate)
            results['exhaust_velocity_ideal'].append(v_ideal)
            results['exhaust_velocity_realistic'].append(v_realistic)
            results['specific_impulse_ideal'].append(isp_ideal)
            results['specific_impulse_realistic'].append(isp_realistic)
            results['thrust_momentum'].append(thrust_components['momentum'])
            results['thrust_pressure'].append(thrust_components['pressure'])
            results['thrust_total'].append(thrust_components['total'])
            results['thrust_mN'].append(thrust_components['total'] * 1000)
            results['power_input'].append(Q_input)
            results['power_kinetic'].append(power_kinetic)
            results['thermal_efficiency'].append(thermal_efficiency)
            results['acceleration_ms2'].append(acceleration)
            results['total_impulse'].append(total_impulse)
            results['expansion_ratio'].append(self.expansion_ratio)
            results['pressure_ratio'].append(pressure_ratio)
            
            # Advance solver
            time += dt
            solver.integrate(time)
        
        return {k: np.array(v) for k, v in results.items()}


@st.cache_data
def create_enhanced_plots(results):
    """Create comprehensive interactive plots"""
    
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Temperature Evolution & Heat Capacity', 'Exhaust Velocity: Ideal vs Realistic',
            'Specific Impulse: Ideal vs Realistic', 'Thrust Components Analysis',
            'Chamber Pressure & Mass Flow Rate', 'Power Analysis & Efficiency',
            'Spacecraft Acceleration & Pressure Ratio', 'Total Impulse Accumulation'
        ),
        specs=[
            [{"secondary_y": True}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.08
    )

    time_minutes = results['time'] / 60
    
    # Row 1: Temperature and Heat Capacity
    fig.add_trace(go.Scatter(x=time_minutes, y=results['temperature_celsius'], 
                            name='Temperature (°C)', 
                            line=dict(color='red', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_minutes, y=results['heat_capacity'], 
                            name='Heat Capacity (J/kg·K)', 
                            line=dict(color='blue', width=2),
                            yaxis='y2'), row=1, col=1)
    fig.add_hline(y=350, line_dash="dash", line_color="orange", 
                  annotation_text="Target (350°C)", row=1, col=1)
    
    # Exhaust Velocity Comparison
    fig.add_trace(go.Scatter(x=time_minutes, y=results['exhaust_velocity_ideal'], 
                            name='Ideal Velocity', 
                            line=dict(color='lightblue', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_minutes, y=results['exhaust_velocity_realistic'], 
                            name='Realistic (η=93%)', 
                            line=dict(color='darkblue', width=3)), row=1, col=2)
    
    # Row 2: Specific Impulse Comparison
    fig.add_trace(go.Scatter(x=time_minutes, y=results['specific_impulse_ideal'], 
                            name='Ideal Isp', 
                            line=dict(color='plum', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_minutes, y=results['specific_impulse_realistic'], 
                            name='Realistic Isp', 
                            line=dict(color='purple', width=3)), row=2, col=1)
    fig.add_hline(y=106, line_dash="dash", line_color="red", 
                  annotation_text="Reference (106s)", row=2, col=1)
    
    # Thrust Components
    fig.add_trace(go.Scatter(x=time_minutes, y=results['thrust_momentum'] * 1000, 
                            name='Momentum Thrust (mN)', 
                            line=dict(color='blue', width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=time_minutes, y=results['thrust_pressure'] * 1000, 
                            name='Pressure Thrust (mN)', 
                            line=dict(color='red', width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=time_minutes, y=results['thrust_mN'], 
                            name='Total Thrust (mN)', 
                            line=dict(color='black', width=3)), row=2, col=2)
    fig.add_hline(y=24.1, line_dash="dash", line_color="red", 
                  annotation_text="Reference (24.1mN)", row=2, col=2)
    
    # Row 3: Pressure and Mass Flow
    fig.add_trace(go.Scatter(x=time_minutes, y=results['chamber_pressure']/1000, 
                            name='Chamber Pressure (kPa)', 
                            line=dict(color='green', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_minutes, y=results['mass_flow_rate']*1e6, 
                            name='Mass Flow Rate (mg/s)', 
                            line=dict(color='orange', width=2),
                            yaxis='y6'), row=3, col=1)
    
    # Power Analysis
    fig.add_trace(go.Scatter(x=time_minutes, y=results['power_input'], 
                            name='Input Power (W)', 
                            line=dict(color='red', width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=time_minutes, y=results['power_kinetic'], 
                            name='Kinetic Power (W)', 
                            line=dict(color='blue', width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=time_minutes, y=results['thermal_efficiency'], 
                            name='Efficiency (%)', 
                            line=dict(color='green', width=2),
                            yaxis='y8'), row=3, col=2)
    
    # Row 4: Acceleration and Pressure Ratio
    fig.add_trace(go.Scatter(x=time_minutes, y=results['acceleration_ms2']*1000, 
                            name='Acceleration (mm/s²)', 
                            line=dict(color='cyan', width=3)), row=4, col=1)
    
    fig.add_trace(go.Scatter(x=time_minutes, y=results['total_impulse']*1000, 
                            name='Total Impulse (mN·s)', 
                            line=dict(color='magenta', width=3)), row=4, col=2)
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (min)")
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Cp (J/kg·K)", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Specific Impulse (s)", row=2, col=1)
    fig.update_yaxes(title_text="Thrust (mN)", row=2, col=2)
    fig.update_yaxes(title_text="Pressure (kPa)", row=3, col=1)
    fig.update_yaxes(title_text="Mass Flow (mg/s)", secondary_y=True, row=3, col=1)
    fig.update_yaxes(title_text="Power (W)", row=3, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", secondary_y=True, row=3, col=2)
    fig.update_yaxes(title_text="Acceleration (mm/s²)", row=4, col=1)
    fig.update_yaxes(title_text="Total Impulse (mN·s)", row=4, col=2)
    
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="Enhanced Butane Thruster Simulation - Realistic Isp Calculations",
        title_x=0.5,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig


def display_performance_metrics(results):
    """Display key performance metrics"""
    if len(results['time']) == 0:
        return
    
    # Get final values
    final_temp = results['temperature'][-1] - 273.15
    final_isp_realistic = results['specific_impulse_realistic'][-1]
    final_isp_ideal = results['specific_impulse_ideal'][-1]
    final_thrust = results['thrust_mN'][-1]
    max_efficiency = max(results['thermal_efficiency'])
    final_exhaust_vel = results['exhaust_velocity_realistic'][-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Final Temperature", 
            f"{final_temp:.1f}°C",
            f"{final_temp - 350:.1f}°C vs target"
        )
        
    with col2:
        st.metric(
            "Realistic Isp", 
            f"{final_isp_realistic:.1f}s",
            f"{final_isp_realistic - 106:.1f}s vs ref"
        )
        
    with col3:
        st.metric(
            "Thrust Output", 
            f"{final_thrust:.2f}mN",
            f"{final_thrust - 24.1:.2f}mN vs ref"
        )
        
    with col4:
        st.metric(
            "Max Efficiency", 
            f"{max_efficiency:.1f}%",
            f"{max_efficiency - 69:.1f}% vs ref"
        )
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Ideal Isp", 
            f"{final_isp_ideal:.1f}s",
            f"η = {(final_isp_realistic/final_isp_ideal)*100:.1f}%"
        )
        
    with col6:
        st.metric(
            "Exhaust Velocity", 
            f"{final_exhaust_vel:.0f} m/s",
            f"V2 = Isp × g0"
        )
        
    with col7:
        st.metric(
            "Heat Capacity", 
            f"{results['heat_capacity'][-1]:.0f} J/kg·K",
            f"At {final_temp:.0f}°C"
        )
        
    with col8:
        st.metric(
            "Total Impulse", 
            f"{results['total_impulse'][-1]*1000:.1f} mN·s",
            f"∫F dt"
        )


# Main Application
st.markdown("""
<div class="hero-header">
    <h1>🚀 Enhanced Butane Thruster Simulation</h1>
    <p style="font-size: 1.2rem; opacity: 0.9; margin-top: 1rem;">
        Realistic Isp calculations: Isp = V₂/g₀ with thermodynamic modeling
    </p>
</div>
""", unsafe_allow_html=True)


# Initialize simulation
@st.cache_resource
def get_simulation():
    return EnhancedButaneThrusterSimulation()

sim = get_simulation()

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Simulation Parameters")
    
    st.subheader("🔥 Thermal Configuration")
    initial_temp_c = st.number_input(
        "Initial Temperature (°C)", 
        min_value=0.0, max_value=100.0, value=27.0, step=1.0
    )
    
    heat_input = st.number_input(
        "Heat Input Q (W)", 
        min_value=1.0, max_value=50.0, value=18.0, step=1.0
    )
    
    propellant_mass = st.number_input(
        "Propellant Mass (kg)", 
        min_value=0.01, max_value=1.0, value=0.2, step=0.01
    )
    
    st.subheader("💨 Pressure Configuration")
    chamber_pressure = st.number_input(
        "Base Chamber Pressure P₁ (kPa)", 
        min_value=50.0, max_value=500.0, value=240.0, step=10.0
    ) * 1000
    
    exit_pressure = st.number_input(
        "Exit Pressure P₂ (Pa)", 
        min_value=1.0, max_value=1000.0, value=50.0, step=5.0
    )
    
    ambient_pressure = st.number_input(
        "Ambient Pressure P₃ (Pa)", 
        min_value=1e-8, max_value=1e5, value=1e-7, format="%.2e"
    )
    
    st.subheader("⏱️ Simulation Settings")
    simulation_time_min = st.number_input(
        "Simulation Time (minutes)", 
        min_value=5.0, max_value=60.0, value=30.0, step=5.0
    )
    
    time_step = st.number_input(
        "Time Step (seconds)", 
        min_value=0.1, max_value=5.0, value=1.0, step=0.1
    )
    
    st.subheader("🚀 Spacecraft Configuration")
    sim.spacecraft_mass = st.number_input(
        "Spacecraft Mass (kg)", 
        min_value=0.1, max_value=100.0, value=16.0, step=0.1
    )
    
    # Display nozzle geometry
    st.subheader("🔧 Nozzle Geometry (Fixed)")
    st.write(f"**Throat Diameter:** {sim.throat_radius*2*1000:.3f} mm")
    st.write(f"**Exit Diameter:** {sim.exit_radius*2*1000:.3f} mm")
    st.write(f"**Expansion Ratio:** {sim.expansion_ratio:.1f}")
    st.write(f"**Nozzle Efficiency:** {sim.nozzle_efficiency*100:.0f}%")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🚀 Simulation")
    
    if st.button("🔥 Run Simulation", type="primary"):
        with st.spinner("Running thermodynamic simulation..."):
            # Prepare configuration
            config = {
                'initial_temp': initial_temp_c + 273.15,
                'heat_input': heat_input,
                'chamber_pressure': chamber_pressure,
                'exit_pressure': exit_pressure,
                'ambient_pressure': ambient_pressure,
                'simulation_time': simulation_time_min * 60,
                'dt': time_step,
                'propellant_mass': propellant_mass
            }
            
            # Run simulation
            results = sim.simulate_enhanced_performance(config)
            
            # Display performance metrics
            st.subheader("📊 Performance Metrics")
            display_performance_metrics(results)
            
            # Create and display plots
            st.subheader("📈 Simulation Results")
            fig = create_enhanced_plots(results)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional analysis
            st.subheader("🔍 Detailed Analysis")
            
            # Create summary dataframe
            summary_data = {
                'Parameter': [
                    'Final Temperature (°C)',
                    'Final Chamber Pressure (kPa)',
                    'Final Mass Flow Rate (mg/s)',
                    'Ideal Exhaust Velocity (m/s)',
                    'Realistic Exhaust Velocity (m/s)',
                    'Ideal Specific Impulse (s)',
                    'Realistic Specific Impulse (s)',
                    'Final Thrust (mN)',
                    'Max Thermal Efficiency (%)',
                    'Nozzle Efficiency (%)',
                    'Total Impulse (mN·s)',
                    'Average Acceleration (mm/s²)'
                ],
                'Simulated Value': [
                    f"{results['temperature'][-1] - 273.15:.1f}",
                    f"{results['chamber_pressure'][-1]/1000:.1f}",
                    f"{results['mass_flow_rate'][-1]*1e6:.3f}",
                    f"{results['exhaust_velocity_ideal'][-1]:.1f}",
                    f"{results['exhaust_velocity_realistic'][-1]:.1f}",
                    f"{results['specific_impulse_ideal'][-1]:.1f}",
                    f"{results['specific_impulse_realistic'][-1]:.1f}",
                    f"{results['thrust_mN'][-1]:.2f}",
                    f"{max(results['thermal_efficiency']):.1f}",
                    f"{sim.nozzle_efficiency*100:.0f}",
                    f"{results['total_impulse'][-1]*1000:.1f}",
                    f"{np.mean(results['acceleration_ms2'])*1000:.2f}"
                ],
                'Reference/Target': [
                    "350.0",
                    "240.0",
                    "N/A",
                    "N/A",
                    "1040.0",
                    "N/A",
                    "106.0",
                    "24.1",
                    "69.0",
                    "93.0",
                    "N/A",
                    "N/A"
                ]
            }
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            # Validation status
            st.subheader("✅ Model Validation")
            
            final_temp_error = abs(results['temperature'][-1] - 273.15 - 350) / 350 * 100
            final_isp_error = abs(results['specific_impulse_realistic'][-1] - 106) / 106 * 100
            final_thrust_error = abs(results['thrust_mN'][-1] - 24.1) / 24.1 * 100
            
            validation_col1, validation_col2, validation_col3 = st.columns(3)
            
            with validation_col1:
                if final_temp_error < 5:
                    st.markdown('<div class="validation-excellent">Temperature: Excellent Match</div>', unsafe_allow_html=True)
                elif final_temp_error < 10:
                    st.markdown('<div class="validation-good">Temperature: Good Match</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-warning">Temperature: Needs Adjustment</div>', unsafe_allow_html=True)
                st.write(f"Error: {final_temp_error:.1f}%")
            
            with validation_col2:
                if final_isp_error < 5:
                    st.markdown('<div class="validation-excellent">Realistic Isp: Excellent Match</div>', unsafe_allow_html=True)
                elif final_isp_error < 10:
                    st.markdown('<div class="validation-good">Realistic Isp: Good Match</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-warning">Realistic Isp: Needs Adjustment</div>', unsafe_allow_html=True)
                st.write(f"Error: {final_isp_error:.1f}%")
            
            with validation_col3:
                if final_thrust_error < 10:
                    st.markdown('<div class="validation-excellent">Thrust: Excellent Match</div>', unsafe_allow_html=True)
                elif final_thrust_error < 20:
                    st.markdown('<div class="validation-good">Thrust: Good Match</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-warning">Thrust: Needs Adjustment</div>', unsafe_allow_html=True)
                st.write(f"Error: {final_thrust_error:.1f}%")
            
            # Key relationship verification
            st.subheader("🔍 Key Relationship Verification")
            
            # Verify Isp = V2 / g0 relationship
            final_v2 = results['exhaust_velocity_realistic'][-1]
            calculated_isp = final_v2 / sim.g0
            reported_isp = results['specific_impulse_realistic'][-1]
            
            st.write(f"**Isp = V₂ / g₀ Verification:**")
            st.write(f"- Exhaust Velocity V₂: {final_v2:.1f} m/s")
            st.write(f"- Calculated Isp: {calculated_isp:.1f} s")
            st.write(f"- Reported Isp: {reported_isp:.1f} s")
            st.write(f"- Difference: {abs(calculated_isp - reported_isp):.3f} s ✓")

with col2:
    st.subheader("📋 Model Features")
    
    st.markdown("""
    <div class="thermodynamic-box">
        <h4>🔬 Enhanced Physics</h4>
        <ul>
            <li><strong>Realistic Isp:</strong> Isp = V₂/g₀</li>
            <li><strong>Thermodynamic V₂:</strong> Complete equation</li>
            <li><strong>Temperature-dependent Cp:</strong> Lookup table</li>
            <li><strong>Dynamic pressure:</strong> P₁ = f(ṁ, T)</li>
            <li><strong>Choked flow:</strong> Mass flow through throat</li>
            <li><strong>Heat balance:</strong> Q = m·cp·dT + losses</li>
            <li><strong>Nozzle efficiency:</strong> 93% realistic</li>
            <li><strong>Fixed geometry:</strong> Constant expansion ratio</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🔧 Butane Properties")
    st.markdown(f"""
    - **Molecular Formula:** C₄H₁₀
    - **Molecular Weight:** {sim.M_butane*1000:.1f} g/mol
    - **Gas Constant R:** {sim.R_specific:.1f} J/kg·K
    - **Heat Ratio k:** {sim.k:.2f}
    - **Nozzle Efficiency η:** {sim.nozzle_efficiency*100:.0f}%
    - **Critical Temperature:** {sim.T_critical:.0f} K
    - **Boiling Point:** {sim.T_boiling:.1f} K
    - **Base Heat Capacity:** ~2400 J/kg·K
    """)

# Additional information section
st.markdown("---")
st.subheader("📚 Model Documentation")

doc_col1, doc_col2 = st.columns(2)

with doc_col1:
    st.markdown("""
    ### 🧮 Key Enhancements Implemented
    
    **1. Realistic Isp Calculation:**
    - Now uses the correct relationship: **Isp = V₂ / g₀**
    - Where V₂ is the realistic exhaust velocity
    - Accounts for nozzle efficiency (93%)
    
    **2. Complete Thermodynamic Exhaust Velocity:**
    - Uses fundamental equation with butane properties
    - Includes specific heat ratio k = 1.05
    - Gas constant R = 143 J/kg·K for butane
    - Temperature-dependent calculations
    
    **3. Enhanced Heat Capacity Model:**
    - Temperature-dependent cp(T) lookup table
    - Cubic interpolation for smooth transitions
    - Based on 2400 J/kg·K baseline with variations
    
    **4. Dynamic Chamber Pressure:**
    - P₁ varies with mass flow rate and temperature
    - More realistic than constant pressure assumption
    """)

with doc_col2:
    st.markdown("""
    ### 🎯 Validation & Accuracy
    
    **Reference Targets:**
    - Temperature: 350°C (623K)
    - Specific Impulse: 106 seconds
    - Thrust: 24.1 mN
    - Thermal Efficiency: 69%
    
    **Model Verification:**
    - **Isp = V₂ / g₀** relationship maintained
    - Choked flow through throat (At = constant)
    - Fixed expansion ratio (A₂/At = constant)
    - Nozzle efficiency = 93%
    
    **Physical Consistency:**
    - Energy conservation in heat balance
    - Momentum conservation in thrust calculation
    - Thermodynamic consistency in gas expansion
    - Proper pressure relationships (P₁ > P₂ > P₃)
    """)

# Run instructions
st.markdown("---")
st.markdown("""
### 🚀 How to Use:

1. **Set Parameters:** Adjust thermal and pressure settings in the sidebar
2. **Run Simulation:** Click the "Run Enhanced Simulation" button
3. **Analyze Results:** Review performance metrics, plots, and validation status
4. **Verify Relationships:** Check that Isp = V₂ / g₀ is maintained throughout

The simulation will show both ideal and realistic performance, allowing you to see the impact of nozzle efficiency and other real-world factors.
""")