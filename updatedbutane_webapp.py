import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Constants
class Constants:
    R = 8.314  # J/(mol·K)
    M_BUTANE = 58.1222e-3  # kg/mol
    T_CRITICAL = 425.13  # K
    P_CRITICAL = 3.796e6  # Pa
    T_BOILING = 272.65  # K
    T_TRIPLE = 134.87  # K
    H_VAP_NORMAL = 22.389e3  # J/mol at 272.05 K
    H_VAP_NORMAL_KG = H_VAP_NORMAL / M_BUTANE  # J/kg
    C_P_LIQUID = 2.4e3  # J/(kg·K)
    GAMMA = 1.10
    G0 = 9.80665  # Standard gravity (m/s²)
    
    # Antoine equation parameters for butane
    ANTOINE_A = 4.70812
    ANTOINE_B = 1200.475
    ANTOINE_C = -13.013

    # Butane combustion properties
    ADIABATIC_FLAME_TEMP = 2240  # K (approximate for butane/air)
    COMBUSTION_EFFICIENCY = 0.95  # Typical combustion efficiency

# Physics calculations with enhanced temperature-dependent Isp
class PhysicsCalculations:
    @staticmethod
    def calculate_combustion_temperature(pressure: float, initial_temp: float) -> float:
        """
        Calculate effective combustion temperature based on pressure and initial temperature
        using a model that accounts for incomplete combustion at lower temps
        """
        # Base temperature effect (higher initial temp leads to more complete combustion)
        temp_factor = min(1.0, (initial_temp - Constants.T_BOILING) / (Constants.T_CRITICAL - Constants.T_BOILING))
        
        # Pressure effect (higher pressure leads to more complete combustion)
        pressure_factor = min(1.0, pressure / Constants.P_CRITICAL)
        
        # Effective combustion temperature
        return Constants.ADIABATIC_FLAME_TEMP * temp_factor * pressure_factor * Constants.COMBUSTION_EFFICIENCY

    @staticmethod
    def calculate_enhanced_isp(
        p_chamber: float,
        T_chamber: float,
        p_ambient: float,
        gamma: float,
        R_specific: float,
        nozzle_efficiency: float = 1.0,
        c_star_efficiency: float = 0.95
    ) -> Dict:
        """
        Specific impulse calculation with temperature-dependent efficiency
        
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
                'expansion_ratio': 1.0,
                'combustion_efficiency': 0
            }
        
        # Calculate effective combustion temperature
        T_combustion = PhysicsCalculations.calculate_combustion_temperature(p_chamber, T_chamber)
        
        # Pressure ratio
        pressure_ratio = p_ambient / p_chamber
        
        # Critical pressure ratio for choked flow
        gamma_term = gamma / (gamma - 1)
        p_crit_ratio = (2 / (gamma + 1)) ** gamma_term
        
        # Check if flow is choked
        is_choked = pressure_ratio <= p_crit_ratio
        
        # Calculate characteristic velocity (c*) with temperature dependence
        c_star_ideal = math.sqrt(gamma * R_specific * T_combustion) / gamma * \
                       ((gamma + 1) / 2) ** ((gamma + 1) / (2 * (gamma - 1)))
        c_star_actual = c_star_ideal * c_star_efficiency
        
        if is_choked:
            # Choked flow conditions
            p_exit = p_chamber * p_crit_ratio
            T_exit = T_combustion * (p_exit / p_chamber) ** ((gamma - 1) / gamma)

            # Ideal exit velocity for choked flow
            v_exit_ideal = math.sqrt(gamma * R_specific * T_exit)
            
            # Thrust coefficient for choked flow
            c_f_ideal = math.sqrt(gamma * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))
        else:
            # Non-choked (overexpanded or underexpanded) flow
            p_exit = p_ambient
            T_exit = T_combustion * (pressure_ratio ** ((gamma - 1) / gamma))
            
            # Ideal exit velocity for non-choked flow
            v_exit_ideal = math.sqrt(2 * gamma * R_specific * T_combustion / (gamma - 1) * 
                                    (1 - pressure_ratio ** ((gamma - 1) / gamma)))
            
            # Thrust coefficient for non-choked flow
            c_f_ideal = math.sqrt(2 * gamma**2 / (gamma - 1) * 
                                 (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)) * 
                                 (1 - pressure_ratio ** ((gamma - 1) / gamma)))
        
        # Apply nozzle efficiency to exit velocity
        v_exit_actual = v_exit_ideal * math.sqrt(nozzle_efficiency)
        c_f_actual = c_f_ideal * math.sqrt(nozzle_efficiency)
        
        # Calculate specific impulse with temperature effects
        isp_theoretical = v_exit_ideal / Constants.G0
        isp_actual = v_exit_actual / Constants.G0
        
        # Calculate expansion ratio (Area_exit / Area_throat)
        if is_choked and p_exit > 0:
            expansion_ratio = (1 / pressure_ratio) * \
                             ((2 + (gamma - 1) * (p_exit / p_chamber)) / (gamma + 1)) ** \
                             ((gamma + 1) / (2 * (gamma - 1)))
        else:
            expansion_ratio = 1.0
        
        # Calculate combustion efficiency based on temperature
        combustion_efficiency = min(1.0, T_combustion / Constants.ADIABATIC_FLAME_TEMP)
        
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
            'expansion_ratio': expansion_ratio,
            'combustion_efficiency': combustion_efficiency,
            'combustion_temperature': T_combustion
        }

# Simulation engine with temperature effects
class SimulationEngine:
    @staticmethod
    def simulate_butane_tank_isp(params: Dict) -> Dict:
        """Simulation with temperature-dependent Isp calculations"""
        # Initialize variables
        T_liquid = params['initial_temp_c'] + 273.15  # K
        T_tank = params['tank_initial_temp_c'] + 273.15  # K
        m_liquid = params['initial_mass']
        R_specific = Constants.R / Constants.M_BUTANE
        integral_error = 0
        
        # Tank material properties
        material_props = TANK_MATERIALS[params['tank_material']]
        tank_heat_capacity = material_props["heat_capacity"]  # J/(kg·K)
        
        # Calculate initial orifice area
        P_initial = PhysicsCalculations.antoine_vapor_pressure(T_liquid)
        A_orifice = PhysicsCalculations.calculate_orifice_area(
            params['target_flow_rate'], P_initial, R_specific, T_liquid, Constants.GAMMA
        )
        
        # Calculate target pressure if in constant Isp mode
        if params['isp_mode'] == "Constant" and params['target_isp'] > 0:
            target_pressure = PhysicsCalculations.calculate_target_pressure_from_isp(
                params['target_isp'], T_liquid, params['ambient_pressure'], Constants.GAMMA, R_specific
            )
        else:
            target_pressure = 0
        
        # Initialize result storage with additional temperature-dependent metrics
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
            'heat_added_heater': [params['base_heater_power']],
            'heat_net': [params['base_heater_power']],
            'heater_status': [1.0 if params['base_heater_power'] > 0 else 0.0],
            # Enhanced thrust parameters with temperature effects
            'specific_impulse': [0],
            'theoretical_isp': [0],
            'exit_velocity': [0],
            'thrust_force': [0],
            'thrust_coefficient': [0],
            'characteristic_velocity': [0],
            'exit_pressure': [params['ambient_pressure'] / 1e5],
            'exit_temperature': [T_liquid - 273.15],
            'momentum_thrust': [0],
            'pressure_thrust': [0],
            'is_choked': [False],
            'pressure_ratio': [1.0],
            'expansion_ratio': [1.0],
            'nozzle_efficiency': [params['nozzle_efficiency']],
            'actual_heater_power': [params['base_heater_power']],
            'isp_error': [0],
            'pressure_error': [0],
            'integral_error': [0],
            # New temperature-dependent metrics
            'combustion_efficiency': [0],
            'combustion_temperature': [T_liquid - 273.15]
        }
        
        t = 0
        
        while (t < params['max_time'] and m_liquid > 0.001 and 
               T_liquid > Constants.T_TRIPLE and T_liquid < Constants.T_CRITICAL):
            # Calculate vapor pressure
            P_vapor = PhysicsCalculations.antoine_vapor_pressure(T_liquid)
            if P_vapor <= 0:
                break
            
            # Calculate actual mass flow rate
            m_dot_actual = PhysicsCalculations.calculate_mass_flow_from_area(
                A_orifice, P_vapor, R_specific, T_liquid, Constants.GAMMA
            )
            m_dot_actual = min(m_dot_actual, m_liquid / params['dt'])
            
            # Calculate thrust parameters with enhanced temperature-dependent Isp
            thrust_params = PhysicsCalculations.calculate_enhanced_isp(
                p_chamber=P_vapor,
                T_chamber=T_liquid,
                p_ambient=params['ambient_pressure'],
                gamma=Constants.GAMMA,
                R_specific=R_specific,
                nozzle_efficiency=params['nozzle_efficiency'],
                c_star_efficiency=params['c_star_efficiency']
            )
            
            # Determine heater power based on Isp mode
            if params['isp_mode'] == "Constant" and params['target_isp'] > 0:
                # Calculate required heater power for constant pressure (and thus Isp)
                C_total = m_liquid * Constants.C_P_LIQUID + params['tank_mass'] * tank_heat_capacity
                current_heater_power, integral_error = PhysicsCalculations.calculate_constant_isp_heater_power(
                    target_pressure=target_pressure,
                    current_pressure=P_vapor,
                    current_temp=T_liquid,
                    mass_flow_rate=m_dot_actual,
                    current_heat_capacity=C_total,
                    base_heater_power=params['base_heater_power'],
                    Kp=params['Kp'],
                    Ki=params['Ki'],
                    integral_error=integral_error
                )
            else:
                # Use base heater power (variable Isp mode)
                current_heater_power = params['base_heater_power']
                integral_error = 0
            
            # Calculate thrust forces
            F_total, F_momentum, F_pressure = PhysicsCalculations.calculate_thrust_force(
                mass_flow_rate=m_dot_actual,
                exit_velocity=thrust_params['exit_velocity'],
                exit_pressure=thrust_params['exit_pressure'],
                ambient_pressure=params['ambient_pressure'],
                exit_area=params['nozzle_exit_area']
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
            C_tank = params['tank_mass'] * tank_heat_capacity  # J/K
            C_total = C_liquid + C_tank  # J/K
            
            # Temperature change (fully insulated system)
            if C_total > 0:
                dT = (Q_net * params['dt']) / C_total  # K
                T_liquid += dT
                T_tank += dT  # Assume thermal equilibrium
            
            # Update mass
            m_liquid -= m_dot_actual * params['dt']
            
            # Calculate Isp error for constant mode
            if params['isp_mode'] == "Constant" and params['target_isp'] > 0:
                isp_error = params['target_isp'] - thrust_params['actual_isp']
                pressure_error_val = target_pressure - P_vapor
            else:
                isp_error = 0
                pressure_error_val = 0
            
            # Advance time
            t += params['dt']
            
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
            
            # Enhanced thrust results with temperature effects
            results['specific_impulse'].append(thrust_params['actual_isp'])
            results['theoretical_isp'].append(thrust_params['theoretical_isp'])
            results['exit_velocity'].append(thrust_params['exit_velocity'])
            results['thrust_force'].append(F_total * 1000)  # Convert to mN
            results['thrust_coefficient'].append(thrust_params['c_f'])
            results['characteristic_velocity'].append(thrust_params['c_star'])
            results['exit_pressure'].append(thrust_params['exit_pressure'] / 1e5)  # bar
            results['exit_temperature'].append(thrust_params['exit_temperature'] - 273.15)  # °C
            results['momentum_thrust'].append(F_momentum * 1000)  # mN
            results['pressure_thrust'].append(F_pressure * 1000)  # mN
            results['is_choked'].append(thrust_params['is_choked'])
            results['pressure_ratio'].append(thrust_params['pressure_ratio'])
            results['expansion_ratio'].append(thrust_params['expansion_ratio'])
            results['nozzle_efficiency'].append(params['nozzle_efficiency'])
            results['actual_heater_power'].append(current_heater_power)
            results['isp_error'].append(isp_error)
            results['pressure_error'].append(pressure_error_val / 1e5)  # bar
            results['integral_error'].append(integral_error / 1e5)  # bar
            results['combustion_efficiency'].append(thrust_params['combustion_efficiency'])
            results['combustion_temperature'].append(thrust_params['combustion_temperature'] - 273.15)  # °C
            
            # Safety checks
            if T_liquid < Constants.T_TRIPLE or T_liquid > Constants.T_CRITICAL:
                break
        
        return {k: np.array(v) for k, v in results.items()}

# Visualization with additional temperature-dependent metrics
def create_enhanced_plots_with_isp(results: Dict, isp_mode: str, target_isp: float) -> go.Figure:
    """Create enhanced interactive Plotly plots with temperature-dependent Isp analysis"""
    # Create subplots with 4x3 layout
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=('Temperature Evolution', 'Pressure Evolution', 'Mass Depletion',
                       'Specific Impulse Analysis', 'Thrust Force Components', 'Exit Velocity & Efficiency',
                       'Heat Transfer Analysis', 'Flow Characteristics', 'Combustion Efficiency',
                       'Nozzle Performance', 'Combustion Temperature', 'Advanced Diagnostics'),
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
    
    # Combustion efficiency
    fig.add_trace(go.Scatter(x=time_hours, y=results['combustion_efficiency'], 
                  name='Combustion Efficiency', line=dict(color='orange', width=2)), row=3, col=3)
    
    # Row 4: Advanced diagnostics
    fig.add_trace(go.Scatter(x=time_hours, y=results['thrust_coefficient'], 
                  name='Thrust Coeff', line=dict(color='darkgreen', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=time_hours, y=results['expansion_ratio'], 
                  name='Expansion Ratio', line=dict(color='orange', width=1)), row=4, col=1)
    
    # Combustion temperature
    fig.add_trace(go.Scatter(x=time_hours, y=results['combustion_temperature'], 
                  name='Combustion Temp', line=dict(color='red', width=2)), row=4, col=2)
    
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
        ('Heat Transfer (W)', 'Flow Rate (μg/s)', 'Combustion Efficiency'),
        ('Thrust Coefficient', 'Combustion Temp (°C)', 'Flow State')
    ]
    
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
        title_text="Butane Tank Simulation with Temperature-Dependent Isp",
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

# UI and main application (same as before, but with new visualization)
def main():
    # ... [previous UI code remains the same until results display]
    
    # Display results if available
    if 'simulation_results' in st.session_state and 'simulation_params' in st.session_state:
        results = st.session_state.simulation_results
        params = st.session_state.simulation_params
        
        # Create and display enhanced plots with temperature effects
        st.subheader("📈 Enhanced Simulation Results with Temperature-Dependent Isp")
        
        fig = create_enhanced_plots_with_isp(
            results, params['isp_mode'], params['target_isp']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance analysis with combustion metrics
        st.subheader("🔥 Combustion Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            avg_comb_temp = np.mean(results['combustion_temperature'][1:])
            max_comb_temp = np.max(results['combustion_temperature'][1:])
            st.markdown(f"""
            <div class="thrust-box">
                <strong>Combustion Temperature:</strong><br>
                Average: {avg_comb_temp:.1f} °C<br>
                Maximum: {max_comb_temp:.1f} °C<br>
                Theoretical Max: {Constants.ADIABATIC_FLAME_TEMP - 273.15:.1f} °C
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_comb_eff = np.mean(results['combustion_efficiency'][1:]) * 100
            st.markdown(f"""
            <div class="isp-box">
                <strong>Combustion Efficiency:</strong><br>
                Average: {avg_comb_eff:.1f}%<br>
                <small>Efficiency depends on temperature and pressure</small>
            </div>
            """, unsafe_allow_html=True)
        
        # ... [rest of the performance analysis and UI remains the same]