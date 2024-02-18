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

    attitude_tethered_hover_analysis(args)
    plt.show()

def attitude_tethered_hover_analysis(args):
    viz = FlightDataVisualizer(args.filename, args.output_folder, 7.3, 17.3)  # 02_AttitudeStabilization 
    viz.set_gains(K_x=[0.05, 0.05, 1.0], 
                  K_v=[0.05, 0.05, 2.0], 
                  K_R=[5.0, 5.0, 1.0, 0.0], 
                  K_W=[10.0, 10.0, 1.5])
    
    # FlightDataVisualizer.x_cut_lines = [0.0, 3.99, 5.3, 7.3, 13.3, 15.3, 20.0]
    # FlightDataVisualizer.x_cut_lines = [0.0, 3.99, 7.3, 9.3, 15.3, 17.3, 22.0]
    FlightDataVisualizer.x_cut_lines = [7.3, 9.3, 15.3, 17.3]

    
    # States (Attitude)
    q, t = viz.read_quaternion_attitude("vehicle_attitude")
    omega, t = viz.read_vector_from_dataset("vehicle_angular_velocity", keys=["xyz"])

    pos, t = viz.read_vector_from_dataset("vehicle_local_position", keys=["x", "y", "z"])
    vel, t = viz.read_vector_from_dataset("vehicle_local_position", keys=["vx", "vy", "vz"])
    try:
        pos_d, t = viz.read_vector_from_dataset("trajectory_setpoint", keys=["position"])
        psi_d, t = viz.read_vector_from_dataset("trajectory_setpoint", keys=["yaw"], num_entry=1)
    except:
        print("No trajectory setpoint is found.")
        pos_d = np.zeros((len(t), 3))
        

    vel_d = vel * 0.0
    omega_d = omega * 0.0

    euler_d = np.zeros([len(t), 3])
    euler_d[:, 2] = psi_d[:, 0]
    print(psi_d)
    q_d = np.zeros([len(t), 4])
    # omega_d = omega * 0.0 + np.array([-0.1, 0.1, -2.5]) # -0.1f, 0.1f, -2.5f
    for i in range(len(t)):
        q_d[i, :] = tf.transformations.quaternion_from_euler(euler_d[i, 0], euler_d[i, 1], euler_d[i, 2])
    
    Plotter.plot_vector3f_batch_separate_with_desire(t, (q, omega), (q_d, omega_d),
                    keys= [r"\mathbf{q}", r"\Omega"], 
                    title= "state_attitude_omega", 
                    limits=[[-0.2, 1.1], [-0.2, 0.2]],
                    legend=[r"$x$", r"$y$", r"$z$"])
    
    # State (Translations)
    Plotter.plot_vector3f_with_desire(t, pos, t, pos_d, [r"\mathbf{x}_{x}", r"\mathbf{x}_{y}", r"\mathbf{x}_{z}"], "local_position", 
                              limits=[[-0.2, 0.2], [-0.5, 0.0], [-1.5, 0.0]])
    
    Plotter.plot_vector3f_batch_separate_with_desire(t, (pos, vel), (pos_d, vel_d),
                    keys= [r"\mathbf{x}_C", r"\mathbf{v}_C"], 
                    title= "state_pos_vel", 
                    limits=[[-1.5, 1.0], [-0.5, 0.5]],
                    legend=[r"$x$", r"$y$", r"$z$"])
    
    # , [-0.04, 0.08], [-0.2, 0.2]
    thrusts, t = viz.read_vector_from_dataset("vehicle_thrust_setpoint", keys=["xyz"])
    torques, t = viz.read_vector_from_dataset("vehicle_torque_setpoint", keys=["xyz"])

    e_x, e_v, err_x, err_v, err_sum = viz.extract_position_error_info(pos, pos_d, vel, vel_d)
    e_R, e_omega, psi, err_omega = viz.extract_attitude_error_info(q, euler_d, omega, omega_d)
    
    # Error States (Translations)
    Plotter.plot_vector3f_batch(t, (e_x, e_v, 10*thrusts), 
                    keys= [r"x", r"y", r"z"], 
                    title= "err_state_pos_vel", 
                    limits=[[-0.5, 1.5], [-1.0, 1.5], [-7.0, 1.0]],
                    marker_style=["-", ":", "--"],
                    legend=[r"$\mathbf{e}_\mathbf{x}$", r"$\mathbf{e}_\mathbf{v}$", r"$10\mathbf{u}_{fd}$"])
    
    # Translation Lyapunov Candidates
    Plotter.plot_vector3f(t, np.array([err_x, err_v, err_sum]).T, 
                    keys= [r"\|\mathbf{e}_\mathbf{x}\|_{\mathbf{K}_\mathbf{x}}", r"\|\mathbf{e}_\mathbf{v}\|_{\mathbf{K}_\mathbf{v}}", r"V_\mathbf{x}"], 
                    title= "lya_pos_vel", limits=[[0, 0.7], [0, 0.3], [0, 1.0]], 
                    colors=["#0072BD", "#0072BD", "#0072BD"])
    
    # Error States (Attitude)
    Plotter.plot_vector3f_batch(t, (e_R, e_omega, 10*torques), 
                    keys= [r"x", r"y", r"z"], 
                    title= "err_state_attitude_omega", 
                    limits=[[-0.2, 0.3], [-0.2, 0.3], [-1.1, 1.0]],
                    marker_style=["-", ":", "--"],
                    legend=[r"$\mathbf{e}_\mathbf{R}$", r"$\mathbf{e}_\Omega$", r"$10\mathbf{u}_{\tau d}$"])
    
    # Attitude Lyapunov Candidate
    Plotter.plot_vector3f(t, np.array([psi, err_omega, psi + err_omega]).T, 
                    keys=[r"\Psi(\mathbf{R}, \mathbf{R}_d)", r"\|\mathbf{e}_{\Omega}\|_{\mathbf{K}_\mathbf{x}}", r"\Psi + \|\mathbf{e}_{\Omega}\|_2"], 
                    title="lya_attitude_omega", 
                    colors=["#0072BD", "#0072BD", "#0072BD"])

    # Control Allocation

    # Allocation Error
    allocated_wrench, t = viz.read_vector_from_dataset("control_allocation_meta_data", keys=["allocated_control"], num_entry=6)
    control_wrench_sp, t = viz.read_vector_from_dataset("control_allocation_meta_data", keys=["control_sp"], num_entry=6)
    err_wrench = allocated_wrench - control_wrench_sp
    Plotter.plot_vector3f_batch_separate(t, (control_wrench_sp[:, :3], allocated_wrench[:, :3], err_wrench[:, :3]), 
                    keys= [r"\mathbf{u}_{\tau d}", r"\mathbf{u}_{\tau }", r"\mathbf{u}_{\tau d}-\mathbf{u}_{\tau }"], 
                    title= "allocated_torques", 
                    limits=[[-0.5, 0.5], [-0.5, 0.5], [0.01, -0.01]],
                    legend=[r"$x$", r"$y$", r"$z$"])
    
    Plotter.plot_vector3f_batch_separate(t, (control_wrench_sp[:, 3:], allocated_wrench[:, 3:], err_wrench[:, 3:]), 
                    keys= [r"\mathbf{u}_{fd}", r"\mathbf{u}_{f}", r"\mathbf{u}_{fd}-\mathbf{u}_{f}"], 
                    title= "allocated_forces", 
                    limits=[[-10, 50], [-10, 50], [0.01, -0.01]],
                    legend=[r"$x$", r"$y$", r"$z$"])
    
    # Allocation Results (motors and servos)
    motor_controls, t = viz.read_vector_from_dataset("actuator_motors", keys=["control"], num_entry=8)
    servo_controls, t = viz.read_vector_from_dataset("actuator_servos", keys=["control"], num_entry=8)
    Plotter.plot_vector3f_batch_separate(t, (motor_controls[:, 0:8:2], motor_controls[:, 1:8:2]), 
                    keys= [r"\mathbf{u}_{\omega_1}", r"\mathbf{u}_{\omega_2}"], 
                    title= "motor_controls", 
                    limits=[[0.5, 0.8], [0.5, 0.8]],
                    colors=None, 
                    legend=["1", "2", "3", "4"],
                    legend_loc="upper right")

    Plotter.plot_vector3f_batch_separate(t, (servo_controls[:, 0:8:2], servo_controls[:, 1:8:2]), 
                    keys= [r"\mathbf{u}_{\eta_x}", r"\mathbf{u}_{\eta_y}"], 
                    title= "servo_controls", 
                    limits=[[-1, 1], [-0.3, 0.4]],
                    colors=None, 
                    legend=["1", "2", "3", "4"],
                    legend_loc="lower right")

if __name__ == '__main__':
    main()