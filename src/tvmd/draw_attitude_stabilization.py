#!/usr/bin/env python3

'''
This script draws the attitude error function and the angular 
velocity error function from the ULog file.

Author: Yen-Cheng Chu (sciyen.ycc@gmail.com)
'''
from Visualizer import FlightDataVisualizer
import Plotter
from matplotlib import pyplot as plt
import argparse
import numpy as np
import tf


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

def main():
    """Command line interface"""

    parser = argparse.ArgumentParser(description='Extract attitude state and attitude setpoints from ULog file and draw the error function.')
    parser.add_argument('filename', metavar='file.ulg', help='ULog input file')
    parser.add_argument('--output_folder', type=str, required=False, default='default', 
                        dest="output_folder", help='figures output folder')
    args = parser.parse_args()

    attitude_stabilization_analysis(args)
    plt.show()

def attitude_stabilization_analysis(args):
    viz = FlightDataVisualizer(args.filename, args.output_folder, 50, 68)  # 02_AttitudeStabilization 
    viz.set_gains(K_x=[0.0, 0.0, 0.5], 
                  K_v=[0.0, 0.0, 2.0], 
                  K_R=[5.0, 5.0, 0.0, 0.0], 
                  K_W=[7.0, 7.0, 0.0])
    
    FlightDataVisualizer.x_cut_lines = [52.15, 56.1, 59.7, 64.6]

    thrusts, t = viz.read_vector_from_dataset("vehicle_thrust_setpoint", keys=["xyz"])
    torques, t = viz.read_vector_from_dataset("vehicle_torque_setpoint", keys=["xyz"])
    q, t = viz.read_quaternion_attitude("vehicle_attitude")
    omega, t = viz.read_vector_from_dataset("vehicle_angular_velocity", keys=["xyz"])
    euler_d = np.zeros([len(t), 3])
    omega_d = omega * 0.0

    e_R, e_omega, psi, err_omega = viz.extract_attitude_error_info(q, euler_d, omega, omega_d)
    att_err_sum = psi + err_omega

    # Attitude Lyapunov Candidate
    Plotter.plot_vector3f(t, np.array([psi, err_omega, att_err_sum]).T, 
                    keys=[r"\Psi(\mathbf{R}, \mathbf{R}_d)", r"\|\mathbf{e}_{\Omega}\|_{\mathbf{K}_x}", r"\Psi + \|\mathbf{e}_{\Omega}\|_2"], 
                    title="attitude_errors", colors=["#0072BD", "#0072BD", "#0072BD"])
    

    # q, omega, torques, thrusts
    Plotter.plot_vector3f_batch_separate(t, (q, omega, torques, thrusts), 
                    keys= [r"\mathbf{q}", r"\Omega", r"\mathbf{u}_{\tau d}", r"\mathbf{u}_{fd}"], 
                    title= "attitude_omega_wrench", 
                    limits=[[-0.2, 1.1], [-0.5, 0.8], [-0.04, 0.08], [-0.2, 0.2]],
                    legend=[r"$x$", r"$y$", r"$z$"])
    
    Plotter.plot_vector3f(t, thrusts, [r"\mathbf{u}_{fx}", r"\mathbf{u}_{fy}", r"\mathbf{u}_{fz}"], "vehicle_thrusts")
    Plotter.plot_vector3f(t, torques, [r"\mathbf{u}_{\tau x}", r"\mathbf{u}_{\tau y}", r"\mathbf{u}_{\tau z}"], "vehicle_torques")

    allocated_wrench, t = viz.read_vector_from_dataset("control_allocation_meta_data", keys=["allocated_control"], num_entry=6)
    control_wrench_sp, t = viz.read_vector_from_dataset("control_allocation_meta_data", keys=["control_sp"], num_entry=6)
    err_wrench = allocated_wrench - control_wrench_sp

    # allocated torque
    Plotter.plot_vector3f_batch_separate(t, (allocated_wrench[:, :3], err_wrench[:, :3]), 
                    keys= [r"\mathbf{u}_{\tau }", r"\mathbf{u}_{\tau d}-\mathbf{u}_{\tau }"], 
                    title= "allocated_torques", 
                    limits=[[-1.0, 1.0], [-0.01, 0.01]],
                    legend=[r"$x$", r"$y$", r"$z$"])
    
    # allocated thrust
    Plotter.plot_vector3f_batch_separate(t, (allocated_wrench[:, 3:], err_wrench[:, 3:]), 
                    keys= [r"\mathbf{u}_{f}", r"\mathbf{u}_{fd}-\mathbf{u}_{f}"], 
                    title= "allocated_forces", 
                    limits=[[-10, 20], [0.01, -0.01]],
                    legend=[r"$x$", r"$y$", r"$z$"])
    
    motor_controls, t = viz.read_vector_from_dataset("actuator_motors", keys=["control"], num_entry=8)
    servo_controls, t = viz.read_vector_from_dataset("actuator_servos", keys=["control"], num_entry=8)
    Plotter.plot_vector3f_batch_separate(t, (motor_controls[:, 0:8:2], motor_controls[:, 1:8:2]), 
                    keys= [r"\mathbf{u}_{\omega_1}", r"\mathbf{u}_{\omega_2}"], 
                    title= "motor_controls", 
                    limits=[[0.08, 0.25], [0.08, 0.25]],
                    colors=None, 
                    legend=["1", "2", "3", "4"])

    Plotter.plot_vector3f_batch_separate(t, (servo_controls[:, 0:8:2], servo_controls[:, 1:8:2]), 
                    keys= [r"\mathbf{u}_{\eta_x}", r"\mathbf{u}_{\eta_y}"], 
                    title= "servo_controls", 
                    limits=[[-1, 1], [-1, 1]],
                    colors=None, 
                    legend=["1", "2", "3", "4"])

if __name__ == '__main__':
    main()