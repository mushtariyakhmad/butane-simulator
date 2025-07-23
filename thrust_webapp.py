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
    page_title=" Butane Tank Thermal Simulation with Pressure-Based Isp Control", 
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
    <h1>🚀 Butane Tank Thermal Simulation with Isp</h1>
    <p style="text-align: center; color: white; margin: 0;">
        Specific impulse calculations with constant Isp control capability
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

# Data classes for organisation 
@dataclass
class TankMaterial:
    name: str 
    heat_capacity: float #J/(kg*K)
    density: float # kg/m^3
    thermal_conductivity: float #W/(m*K)

@dataclass
class ThrustParameters:
    theoretical_isp: float 
    actual_isp: float
    c_star: float
    exit_velocity: float
    exit_pressure: float 
    exit_temmperature: float 
    is_choked: bool 
    pressure_ratio: float 
    expansion_ration: float 

@dataclass
class SimulationParameters:
    initial_temp_c: float  # Initial temperature in Celsius
    initial_mass: float  # Initial mass in kg
    target_flow_rate: float 
    dt: float 
    max_time: float 
    tank_mass: float 
    tank_material: float 
    base_heater_power: float 
    tank_initial_temp_c: float
    nozzle_exit_area: float
    ambient_pressure: float
    isp_mode: str
    target_isp: float
    nozzle_efficiency: float
    c_star_efficiency: float
    Kp: float = 0.1
    Ki: float = 0.01

@dataclass
class SimulationResults:
    time: np.ndarray
    temperature_liquid: np.ndarray
    temperature_tank: np.ndarray
    temperature_combined: np.ndarray
    pressure: np.ndarray
    target_pressure: np.ndarray
    mass: np.ndarray
    flow_rate: np.ndarray
    h_vap: np.ndarray
    heat_removed_vaporization: np.ndarray
    heat_added_heater: np.ndarray
    heat_net: np.ndarray
    heater_status: np.ndarray
    specific_impulse: np.ndarray
    theoretical_isp: np.ndarray
    exit_velocity: np.ndarray
    thrust_force: np.ndarray
    thrust_coefficient: np.ndarray
    characteristic_velocity: np.ndarray
    exit_pressure: np.ndarray
    exit_temperature: np.ndarray
    momentum_thrust: np.ndarray
    pressure_thrust: np.ndarray
    is_choked: np.ndarray
    pressure_ratio: np.ndarray
    expansion_ratio: np.ndarray
    nozzle_efficiency: np.ndarray
    actual_heater_power: np.ndarray
    isp_error: np.ndarray
    pressure_error: np.ndarray
    integral_error: np.ndarray

# Tank material database
TANK_MATERIALS = {
    "Aluminum 6061": TankMaterial(
        name="Aluminum 6061",
        heat_capacity=896,
        density=2700,
        thermal_conductivity=167
    ),
    "Stainless Steel 304": TankMaterial(
        name="Stainless Steel 304",
        heat_capacity=500,
        density=8000,
        thermal_conductivity=16.2
    ),
    "Stainless Steel 316": TankMaterial(
        name="Stainless Steel 316",
        heat_capacity=500,
        density=8000,
        thermal_conductivity=16.3
    ),
    "Stainless Steel 316L": TankMaterial(
        name="Stainless Steel 316L",
        heat_capacity=500,
        density=8000,
        thermal_conductivity=16.3
    ),
    "TC-4 Titanium": TankMaterial(
        name="TC-4 Titanium",
        heat_capacity=523,
        density=4430,
        thermal_conductivity=6.7
    )
}

# Physics calculations
class PhysicsCalculations:
    @staticmethod
    @st.cache_data
    def antoine_vapor_pressure(T: float) -> float:
        """Calculate vapor pressure using Antoine equation with validation"""
        if T < Constants.T_TRIPLE or T > Constants.T_CRITICAL:
            st.warning(f"Temperature {T}K outside valid range for Antoine equation")
            return 0.0
        
        log_p_bar = Constants.ANTOINE_A - Constants.ANTOINE_B / (T + Constants.ANTOINE_C)
        return 10 ** log_p_bar * 1e5  # Convert bar to Pa

    @staticmethod
    @st.cache_data
    def calculate_enhanced_isp(
        p_chamber: float,
        T_chamber: float,
        p_ambient: float,
        gamma: float,
        R_specific: float,
        nozzle_efficiency: float = 1.0,
        c_star_efficiency: float = 0.95
    ) -> ThrustParameters:
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
            return ThrustParameters(
                theoretical_isp=0,
                actual_isp=0,
                c_star=0,
                c_f=0,
                exit_velocity=0,
                exit_pressure=p_ambient,
                exit_temperature=T_chamber,
                is_choked=False,
                pressure_ratio=1.0,
                expansion_ratio=1.0
            )
        
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
            T_exit = T_chamber * (p_exit / p_chamber) ** ((gamma - 1) / gamma)

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
        isp_theoretical = v_exit_ideal / Constants.G0
        isp_actual = v_exit_actual / Constants.G0
        
        # Calculate expansion ratio (Area_exit / Area_throat)
        if is_choked and p_exit > 0:
            # For choked flow, calculate ideal expansion ratio
            expansion_ratio = (1 / pressure_ratio) * \
                             ((2 + (gamma - 1) * (p_exit / p_chamber)) / (gamma + 1)) ** \
                             ((gamma + 1) / (2 * (gamma - 1)))
        else:
            expansion_ratio = 1.0
        
        return ThrustParameters(
            theoretical_isp=isp_theoretical,
            actual_isp=isp_actual,
            c_star=c_star_actual,
            c_f=c_f_actual,
            exit_velocity=v_exit_actual,
            exit_pressure=p_exit,
            exit_temperature=T_exit,
            is_choked=is_choked,
            pressure_ratio=pressure_ratio,
            expansion_ratio=expansion_ratio
        )

    @staticmethod
    @st.cache_data
    def calculate_constant_isp_heater_power(
        target_pressure: float,
        current_pressure: float,
        current_temp: float,
        mass_flow_rate: float,
        current_heat_capacity: float,
        base_heater_power: float = 0,
        Kp: float = 0.1,
        Ki: float = 0.01,
        integral_error: float = 0
    ) -> Tuple[float, float]:
        """
        Calculate required heater power to maintain constant pressure (and thus Isp)
        
        Uses a PI controller to adjust heater power based on pressure error
        
        Parameters:
        - target_pressure: Desired chamber pressure (Pa)
        - current_pressure: Current chamber pressure (Pa)
        - current_temp: Current liquid temperature (K)
        - mass_flow_rate: Current mass flow rate (kg/s)
        - current_heat_capacity: Current system heat capacity (J/K)
        - base_heater_power: Minimum heater power (W)
        - Kp: Proportional gain
        - Ki: Integral gain
        - integral_error: Accumulated integral error
        """
        if target_pressure <= 0 or current_heat_capacity <= 0:
            return base_heater_power, 0
        
        # Calculate pressure error
        error = target_pressure - current_pressure
        
        # Proportional term
        P_term = Kp * error
        
        # Integral term (with anti-windup)
        if abs(integral_error) < 1000:  # Simple anti-windup
            integral_error += error
        
        I_term = Ki * integral_error
        
        # Calculate required temperature change to achieve target pressure
        # Using Antoine equation to estimate required temperature
        if current_pressure > 0:
            # Estimate required temperature change using Antoine equation derivative
            # dP/dT ≈ P * B / (T + C)^2 (from Antoine equation)
            dP_dT = current_pressure * Constants.ANTOINE_B / (current_temp + Constants.ANTOINE_C)**2
            
            if dP_dT > 0:
                # Estimate required temperature change
                delta_T = error / dP_dT
            else:
                delta_T = 0
        else:
            delta_T = 0
        
        # Calculate additional power needed based on temperature change
        additional_power = delta_T * current_heat_capacity * 0.1  # Conservative factor
        
        # Combine terms with base power
        required_power = base_heater_power + additional_power + P_term + I_term
        
        # Limit to reasonable range (0-100W)
        required_power = min(max(required_power, 0), 100)
        
        return required_power, integral_error

    @staticmethod
    @st.cache_data
    def calculate_thrust_force(
        mass_flow_rate: float,
        exit_velocity: float,
        exit_pressure: float,
        ambient_pressure: float,
        exit_area: float
    ) -> Tuple[float, float, float]:
        """Calculate thrust force using momentum and pressure terms"""
        # Momentum thrust
        F_momentum = mass_flow_rate * exit_velocity
        
        # Pressure thrust
        F_pressure = (exit_pressure - ambient_pressure) * exit_area
        
        # Total thrust
        F_total = F_momentum + F_pressure
        
        return F_total, F_momentum, F_pressure

    @staticmethod
    @st.cache_data
    def calculate_target_pressure_from_isp(
        target_isp: float,
        T_chamber: float,
        p_ambient: float,
        gamma: float,
        R_specific: float
    ) -> float:
        """
        Calculate required chamber pressure to achieve target Isp
        
        This is an approximation since Isp depends on both pressure and temperature,
        but we can estimate based on choked flow conditions where:
        Isp ≈ sqrt(gamma * R * T_chamber) / g0 * factor
        """
        if target_isp <= 0:
            return p_ambient
        
        # From choked flow Isp equation, solve for pressure that would give this Isp
        # This is an approximation since pressure affects temperature through vapor pressure curve
        v_exit = target_isp * Constants.G0
        p_guess = p_ambient * math.exp(v_exit / math.sqrt(R_specific * T_chamber))
        
        # Refine using numerical solution
        def pressure_error(p):
            isp_data = PhysicsCalculations.calculate_enhanced_isp(
                p, T_chamber, p_ambient, gamma, R_specific
            )
            return isp_data.actual_isp - target_isp
        
        # Simple secant method to find pressure
        p_low = p_ambient
        p_high = 10 * p_ambient
        tol = 1e3  # 1 kPa tolerance
        
        for _ in range(10):
            p_mid = (p_low + p_high) / 2
            error = pressure_error(p_mid)
            
            if abs(error) < tol:
                break
                
            if error > 0:
                p_high = p_mid
            else:
                p_low = p_mid
        
        return max(p_mid, p_ambient)

    @staticmethod
    @st.cache_data
    def calculate_orifice_area(
        m_dot: float,
        p1: float,
        R_specific: float,
        T1: float,
        k: float
    ) -> float:
        """Calculate required orifice area for choked flow"""
        if p1 <= 0:
            return 0
        denominator = k * (2 / (k + 1)) ** ((k + 1) / (k - 1))
        return (m_dot / p1) * math.sqrt(R_specific * T1 / denominator)

    @staticmethod
    @st.cache_data
    def calculate_mass_flow_from_area(
        A_t: float,
        p1: float,
        R_specific: float,
        T1: float,
        k: float
    ) -> float:
        """Calculate mass flow rate from orifice area with safety checks"""
        if p1 <= 0 or A_t <= 0 or T1 <= 0:
            return 0
        try:
            multiplier = k * (2 / (k + 1)) ** ((k + 1) / (k - 1))
            return A_t * p1 * math.sqrt(multiplier / (R_specific * T1))
        except (ValueError, ZeroDivisionError):
            return 0

    @staticmethod
    @st.cache_data
    def calculate_enthalpy_vaporization(T: float) -> float:
        """Watson correlation for butane - verify exponent"""
        if T >= Constants.T_CRITICAL:
            return 0
        # Use validated correlation for butane
        n = 0.38  # Should be verified from literature
        ratio = (Constants.T_CRITICAL - T) / (Constants.T_CRITICAL - 272.05)
        return max(Constants.H_VAP_NORMAL_KG * (ratio ** n), 0)

# Simulation engine
class SimulationEngine:
    @staticmethod
    @st.cache_data
    def simulate_butane_tank_enhanced_isp(params: SimulationParameters) -> SimulationResults:
        """Enhanced simulation with pressure-based Isp control"""
        # Initialize variables
        T_liquid = params.initial_temp_c + 273.15  # K
        T_tank = params.tank_initial_temp_c + 273.15  # K
        m_liquid = params.initial_mass
        R_specific = Constants.R / Constants.M_BUTANE
        integral_error = 0
        
        # Tank material properties
        material_props = TANK_MATERIALS[params.tank_material]
        tank_heat_capacity = material_props.heat_capacity  # J/(kg·K)
        
        # Calculate initial orifice area
        P_initial = PhysicsCalculations.antoine_vapor_pressure(T_liquid)
        A_orifice = PhysicsCalculations.calculate_orifice_area(
            params.target_flow_rate, P_initial, R_specific, T_liquid, Constants.GAMMA
        )
        
        # Calculate target pressure if in constant Isp mode
        if params.isp_mode == "Constant" and params.target_isp > 0:
            target_pressure = PhysicsCalculations.calculate_target_pressure_from_isp(
                params.target_isp, T_liquid, params.ambient_pressure, Constants.GAMMA, R_specific
            )
        else:
            target_pressure = 0
        
        # Initialize result storage
        results = {
            'time': [0],
            'temperature_liquid': [T_liquid - 273.15],
            'temperature_tank': [T_tank - 273.15],
            'temperature_combined': [T_liquid - 273.15],
            'pressure': [P_initial / 1e5],
            'target_pressure': [target_pressure / 1e5],
            'mass': [m_liquid],
            'flow_rate': [0],
            'h_vap': [PhysicsCalculations.calculate_enthalpy_vaporization(T_liquid) / 1000],
            'heat_removed_vaporization': [0],
            'heat_added_heater': [params.base_heater_power],
            'heat_net': [params.base_heater_power],
            'heater_status': [1.0 if params.base_heater_power > 0 else 0.0],
            # Enhanced thrust parameters
            'specific_impulse': [0],
            'theoretical_isp': [0],
            'exit_velocity': [0],
            'thrust_force': [0],
            'thrust_coefficient': [0],
            'characteristic_velocity': [0],
            'exit_pressure': [params.ambient_pressure / 1e5],
            'exit_temperature': [T_liquid - 273.15],
            'momentum_thrust': [0],
            'pressure_thrust': [0],
            'is_choked': [False],
            'pressure_ratio': [1.0],
            'expansion_ratio': [1.0],
            'nozzle_efficiency': [params.nozzle_efficiency],
            'actual_heater_power': [params.base_heater_power],
            'isp_error': [0],
            'pressure_error': [0],
            'integral_error': [0]
        }
        
        t = 0
        
        while (t < params.max_time and m_liquid > 0.001 and 
               T_liquid > Constants.T_TRIPLE and T_liquid < Constants.T_CRITICAL):
            # Calculate vapor pressure
            P_vapor = PhysicsCalculations.antoine_vapor_pressure(T_liquid)
            if P_vapor <= 0:
                break
            
            # Calculate actual mass flow rate
            m_dot_actual = PhysicsCalculations.calculate_mass_flow_from_area(
                A_orifice, P_vapor, R_specific, T_liquid, Constants.GAMMA
            )
            m_dot_actual = min(m_dot_actual, m_liquid / params.dt)
            
            # Calculate thrust parameters with enhanced Isp
            thrust_params = PhysicsCalculations.calculate_enhanced_isp(
                p_chamber=P_vapor,
                T_chamber=T_liquid,
                p_ambient=params.ambient_pressure,
                gamma=Constants.GAMMA,
                R_specific=R_specific,
                nozzle_efficiency=params.nozzle_efficiency,
                c_star_efficiency=params.c_star_efficiency
            )
            
            # Determine heater power based on Isp mode
            if params.isp_mode == "Constant" and params.target_isp > 0:
                # Calculate required heater power for constant pressure (and thus Isp)
                C_total = m_liquid * Constants.C_P_LIQUID + params.tank_mass * tank_heat_capacity
                current_heater_power, integral_error = PhysicsCalculations.calculate_constant_isp_heater_power(
                    target_pressure=target_pressure,
                    current_pressure=P_vapor,
                    current_temp=T_liquid,
                    mass_flow_rate=m_dot_actual,
                    current_heat_capacity=C_total,
                    base_heater_power=params.base_heater_power,
                    Kp=params.Kp,
                    Ki=params.Ki,
                    integral_error=integral_error
                )
            else:
                # Use base heater power (variable Isp mode)
                current_heater_power = params.base_heater_power
                integral_error = 0
            
            # Calculate thrust forces
            F_total, F_momentum, F_pressure = PhysicsCalculations.calculate_thrust_force(
                mass_flow_rate=m_dot_actual,
                exit_velocity=thrust_params.exit_velocity,
                exit_pressure=thrust_params.exit_pressure,
                ambient_pressure=params.ambient_pressure,
                exit_area=params.nozzle_exit_area
            )
            
            # Calculate enthalpy of vaporization at current temperature
            h_vap_current = PhysicsCalculations.calculate_enthalpy_vaporization(T_liquid)
            
            # Heat removed by vaporization (cooling effect)
            Q_removed_vaporization = m_dot_actual * h_vap_current  # W (J/s)
            
            # Heat added by heater (heating effect)
            Q_added_heater = current_heater_power  # W
            
            # Net heat transfer
            Q_net = Q_added_heater - Q_removed_vaporization  # W
            
            # Combined thermal mass (liquid + tank)
            C_liquid = m_liquid * Constants.C_P_LIQUID  # J/K
            C_tank = params.tank_mass * tank_heat_capacity  # J/K
            C_total = C_liquid + C_tank  # J/K
            
            # Temperature change (fully insulated system)
            if C_total > 0:
                dT = (Q_net * params.dt) / C_total  # K
                T_liquid += dT
                T_tank += dT  # Assume thermal equilibrium
            
            # Update mass
            m_liquid -= m_dot_actual * params.dt
            
            # Calculate Isp error for constant mode
            if params.isp_mode == "Constant" and params.target_isp > 0:
                isp_error = params.target_isp - thrust_params.actual_isp
                pressure_error_val = target_pressure - P_vapor
            else:
                isp_error = 0
                pressure_error_val = 0
            
            # Advance time
            t += params.dt
            
            # Store results
            results['time'].append(t)
            results['temperature_liquid'].append(T_liquid - 273.15)
            results['temperature_tank'].append(T_tank - 273.15)
            results['temperature_combined'].append(T_liquid - 273.15)
            results['pressure'].append(P_vapor / 1e5)
            results['target_pressure'].append(target_pressure / 1e5)
            results['mass'].append(m_liquid)
            results['flow_rate'].append(m_dot_actual * 1e6)  # μg/s
            results['h_vap'].append(h_vap_current / 1000)  # kJ/kg
            results['heat_removed_vaporization'].append(Q_removed_vaporization)
            results['heat_added_heater'].append(Q_added_heater)
            results['heat_net'].append(Q_net)
            results['heater_status'].append(1.0 if current_heater_power > 0 else 0.0)
            
            # Enhanced thrust results
            results['specific_impulse'].append(thrust_params.actual_isp)
            results['theoretical_isp'].append(thrust_params.theoretical_isp)
            results['exit_velocity'].append(thrust_params.exit_velocity)
            results['thrust_force'].append(F_total * 1000)  # Convert to mN
            results['thrust_coefficient'].append(thrust_params.c_f)
            results['characteristic_velocity'].append(thrust_params.c_star)
            results['exit_pressure'].append(thrust_params.exit_pressure / 1e5)  # bar
            results['exit_temperature'].append(thrust_params.exit_temperature - 273.15)  # °C
            results['momentum_thrust'].append(F_momentum * 1000)  # mN
            results['pressure_thrust'].append(F_pressure * 1000)  # mN
            results['is_choked'].append(thrust_params.is_choked)
            results['pressure_ratio'].append(thrust_params.pressure_ratio)
            results['expansion_ratio'].append(thrust_params.expansion_ratio)
            results['nozzle_efficiency'].append(params.nozzle_efficiency)
            results['actual_heater_power'].append(current_heater_power)
            results['isp_error'].append(isp_error)
            results['pressure_error'].append(pressure_error_val / 1e5)  # bar
            results['integral_error'].append(integral_error / 1e5)  # bar
            
            # Safety checks
            if T_liquid < Constants.T_TRIPLE or T_liquid > Constants.T_CRITICAL:
                break
        
        return SimulationResults(**{k: np.array(v) for k, v in results.items()})

# Visualization
class Visualization:
    @staticmethod
    def create_enhanced_plots_with_isp(results: SimulationResults, isp_mode: str, target_isp: float) -> go.Figure:
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
        
        time_hours = results.time / 3600
        
        # Row 1: Basic parameters
        fig.add_trace(go.Scatter(x=time_hours, y=results.temperature_combined, 
                                name='System Temp', line=dict(color='blue', width=2)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=time_hours, y=results.pressure, 
                                name='Pressure', line=dict(color='green', width=2)), row=1, col=2)
        
        fig.add_trace(go.Scatter(x=time_hours, y=results.mass, 
                                name='Mass', line=dict(color='red', width=2)), row=1, col=3)
        
        # Row 2: Enhanced Isp analysis
        fig.add_trace(go.Scatter(x=time_hours, y=results.specific_impulse, 
                                name='Actual Isp', line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_hours, y=results.theoretical_isp, 
                                name='Theoretical Isp', line=dict(color='purple', width=1, dash='dash')), row=2, col=1)
        
        if isp_mode == "Constant" and target_isp > 0:
            fig.add_trace(go.Scatter(x=time_hours, y=[target_isp] * len(time_hours), 
                                name='Target Isp', line=dict(color='red', width=2, dash='dot')), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=time_hours, y=results.thrust_force, 
                                name='Total Thrust', line=dict(color='orange', width=2)), row=2, col=2)
        fig.add_trace(go.Scatter(x=time_hours, y=results.momentum_thrust, 
                                name='Momentum', line=dict(color='green', width=1)), row=2, col=2)
        fig.add_trace(go.Scatter(x=time_hours, y=results.pressure_thrust, 
                                name='Pressure', line=dict(color='red', width=1)), row=2, col=2)
        
        fig.add_trace(go.Scatter(x=time_hours, y=results.exit_velocity, 
                                name='Exit Velocity', line=dict(color='brown', width=2)), row=2, col=3)
        fig.add_trace(go.Scatter(x=time_hours, y=results.nozzle_efficiency * np.max(results.exit_velocity), 
                                name='Nozzle Eff. (scaled)', line=dict(color='cyan', width=1)), row=2, col=3)
        
        # Row 3: Heat transfer and flow
        fig.add_trace(go.Scatter(x=time_hours, y=results.heat_added_heater, 
                                name='Heat Added', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Scatter(x=time_hours, y=-results.heat_removed_vaporization, 
                                name='Heat Removed', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=time_hours, y=results.heat_net, 
                                name='Net Heat', line=dict(color='black', width=3)), row=3, col=1)
        
        fig.add_trace(go.Scatter(x=time_hours, y=results.flow_rate, 
                                name='Flow Rate', line=dict(color='cyan', width=2)), row=3, col=2)
        
        # Isp performance metrics
        if isp_mode == "Constant":
            fig.add_trace(go.Scatter(x=time_hours, y=results.isp_error, 
                                    name='Isp Error', line=dict(color='red', width=2)), row=3, col=3)
        else:
            fig.add_trace(go.Scatter(x=time_hours, y=results.specific_impulse, 
                                    name='Variable Isp', line=dict(color='purple', width=2)), row=3, col=3)
        
        # Row 4: Advanced diagnostics
        fig.add_trace(go.Scatter(x=time_hours, y=results.thrust_coefficient, 
                                name='Thrust Coeff', line=dict(color='darkgreen', width=2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=time_hours, y=results.expansion_ratio, 
                                name='Expansion Ratio', line=dict(color='orange', width=1)), row=4, col=1)
        
        # Control system
        fig.add_trace(go.Scatter(x=time_hours, y=results.actual_heater_power, 
                                name='Actual Power', line=dict(color='red', width=2)), row=4, col=2)
        
        # Choked flow indicator
        choked_indicator = [1 if choked else 0 for choked in results.is_choked]
        fig.add_trace(go.Scatter(x=time_hours, y=choked_indicator, 
                                name='Choked Flow', line=dict(color='purple, width=2, dash='dash')), row=4, col=3)))
