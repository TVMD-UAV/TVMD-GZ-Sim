#!/usr/bin/env python3

'''
This script draws the attitude error function and the angular 
velocity error function from the ULog file.

Author: Yen-Cheng Chu (sciyen.ycc@gmail.com)
'''

from collections import defaultdict
import argparse
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from pyulog import ULog
import numpy as np
import tf

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

def main():
    """Command line interface"""

    parser = argparse.ArgumentParser(description='Extract attitude state and attitude setpoints from ULog file and draw the error function.')
    parser.add_argument('filename', metavar='file.ulg', help='ULog input file')
    args = parser.parse_args()

    viz = FlightDataVisualizer(args.filename, 0, 20)
    t, psi = viz.extract_attitude_error_info()
    _, omega_err = viz.extract_angular_velocity_error_info()
    att_err_sum = psi + omega_err

    plot_vector3f(t, np.array([psi, omega_err, att_err_sum]).T, 
                    keys=[r"\Psi(\mathbf{R}, \mathbf{R}_d)", r"\|\mathbf{e}_{\Omega}\|_{\mathbf{K}_x}", r"\Psi + \|\mathbf{e}_{\Omega}\|_2"], 
                    title="attitude_errors", colors=["#000000", "#000000", "#000000"])

    thrusts, t = viz.read_vector_from_dataset("vehicle_thrust_setpoint", keys=["xyz"])
    plot_vector3f(t, thrusts, [r"\mathbf{u}_{fx}", r"\mathbf{u}_{fy}", r"\mathbf{u}_{fz}"], "vehicle_thrusts")

    torques, t = viz.read_vector_from_dataset("vehicle_torque_setpoint", keys=["xyz"])
    plot_vector3f(t, torques, [r"\mathbf{u}_{\tau x}", r"\mathbf{u}_{\tau y}", r"\mathbf{u}_{\tau z}"], "vehicle_torques")


    pos_d, t = viz.read_vector_from_dataset("trajectory_setpoint", keys=["position"])
    pos, t = viz.read_vector_from_dataset("vehicle_local_position", keys=["x", "y", "z"])
    plot_vector3f_with_desire(t, pos, t, pos_d, [r"\mathbf{x}_{x}", r"\mathbf{x}_{y}", r"\mathbf{x}_{z}"], "local_position", limits=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    
    vel, t = viz.read_vector_from_dataset("vehicle_local_position", keys=["vx", "vy", "vz"])
    vel_d = np.zeros((1, 3))
    Kx = np.diag([0.5, 0.5, 1.0])
    Kv = np.diag([0.5, 0.5, 2.0])
    e_x = (pos - pos_d)
    e_v = (vel - vel_d)
    err_x = np.linalg.norm(e_x @ Kx, axis=1)
    err_v = np.linalg.norm(e_v @ Kv, axis=1)
    err_sum = err_x + err_v
    plot_vector3f(t, np.array([err_x, err_v, err_sum]).T, 
                    keys= [r"\|\mathbf{e}_x\|_{\mathbf{K}_x}", r"\|\mathbf{e}_v\|_{\mathbf{K}_v}", r"V_\mathbf{x}"], 
                    title= "position_errors", limits=[[0, 3.0], [0, 1.0], [0, 3.0]], 
                    colors=["#000000", "#000000", "#000000"])

    plot_vector3f_batch(t, (e_x, e_v, thrusts), 
                    keys= [r"x", r"y", r"z"], 
                    title= "translational_response", 
                    limits=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                    marker_style=["-", "--", "-."],
                    legend=[r"$\mathbf{e}_x$", r"$\mathbf{e}_v$", r"$\mathbf{u}_f$"])

    allocated_wrench, t = viz.read_vector_from_dataset("control_allocation_meta_data", keys=["allocated_control"], num_entry=6)
    control_wrench_sp, t = viz.read_vector_from_dataset("control_allocation_meta_data", keys=["control_sp"], num_entry=6)
    err_wrench = allocated_wrench - control_wrench_sp
    plot_vector3f_batch_separate(t, (control_wrench_sp[:, :3], allocated_wrench[:, :3], err_wrench[:, :3]), 
                    keys= [r"\mathbf{u}_{\tau d}", r"\mathbf{u}_{\tau }", r"\mathbf{u}_{\tau d}-\mathbf{u}_{\tau }"], 
                    title= "allocated_torques", 
                    limits=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                    legend=[r"$x$", r"$y$", r"$z$"])
    
    plot_vector3f_batch_separate(t, (control_wrench_sp[:, 3:], allocated_wrench[:, 3:], err_wrench[:, 3:]), 
                    keys= [r"\mathbf{u}_{fd}", r"\mathbf{u}_{f}", r"\mathbf{u}_{fd}-\mathbf{u}_{f}"], 
                    title= "allocated_forces", 
                    limits=[[-10, 50], [-10, 50], [0.01, -0.01]],
                    legend=[r"$x$", r"$y$", r"$z$"])
    
    motor_controls, t = viz.read_vector_from_dataset("actuator_motors", keys=["control"], num_entry=8)
    servo_controls, t = viz.read_vector_from_dataset("actuator_servos", keys=["control"], num_entry=8)
    plot_vector3f_batch_separate(t, (thrusts, motor_controls[:, 0:8:2], motor_controls[:, 1:8:2]), 
                    keys= [r"\mathbf{u}_{fd}", r"\mathbf{u}_{\omega_1}", r"\mathbf{u}_{\omega_2}"], 
                    title= "motor_controls", 
                    limits=[[-1, 1], [0, 1], [0, 1]],
                    colors=None)

    plot_vector3f_batch_separate(t, (torques, servo_controls[:, 0:8:2], servo_controls[:, 1:8:2]), 
                    keys= [r"\mathbf{u}_{fd}", r"\mathbf{u}_{\eta_x}", r"\mathbf{u}_{\eta_y}"], 
                    title= "servo_controls", 
                    limits=[[-0.3, 0.3], [-1, 1], [-1, 1]],
                    colors=None)

class FlightDataVisualizer:
    def __init__(self, fname, t1, t2) -> None:
        self.t1 = t1
        self.t2 = t2
        self.ulog = ULog(fname)
        self.data = self.ulog.data_list
        self.multiids = defaultdict(set)
        for d in self.data:
            self.multiids[d.name].add(d.multi_id)
            
        # print(data)
        # print(len(data))
        # print(multiids)
            
        # Reference initial timestamp
        self.armed_time = self.ulog.get_dataset("vehicle_status").data["armed_time"][0]
        print("armed_time: ", self.armed_time)

        # Reference timestamp
        self.ref_timestamp = self.ulog.get_dataset("vehicle_angular_velocity").data["timestamp"]
        print("ref_timestamp: ", len(self.ref_timestamp))

        t, d = FlightDataVisualizer._time_vector_clipping(self.armed_time, self.ref_timestamp, self.ref_timestamp, t1, t2)
        self.len_clipped_timestamp = len(t)

    def extract_attitude_error_info(self):
        euler_d, t_d = self._read_attitude_from_dataset("vehicle_attitude_setpoint", type="euler")
        q, t = self._read_attitude_from_dataset("vehicle_attitude", type="quat", quat_name="q")
        q_reset, t = self._read_attitude_from_dataset("vehicle_attitude", type="quat", quat_name="delta_q_reset")
        q_reset = FlightDataVisualizer._trans_wxyz_to_xyzw_batch(q_reset)
        q = FlightDataVisualizer._trans_wxyz_to_xyzw_batch(q)

        psi = np.zeros(len(t))
        for i in range(len(t)):
            R_d = tf.transformations.euler_matrix(euler_d[i, 0], euler_d[i, 1], euler_d[i, 2], 'sxyz')
            q_correct = tf.transformations.quaternion_multiply(q[i, :], q_reset[i, :])
            # print("q: ", q[i, :])
            # print("q_correct: ", q_correct)
            # q_correct = tf.transformations.quaternion_multiply(q_reset[i, :], q[i, :])
            # q_correct[0] = -q_correct[0]
            R = tf.transformations.quaternion_matrix(q_correct)
            K_R = np.diag([5.0, 5.0, 1.0, 0])
            psi[i] = 0.5 * np.trace(K_R @ (np.eye(4) - R_d.T @ R))
        return t, psi
    
    def extract_angular_velocity_error_info(self):
        omega, t = self.read_vector_from_dataset("vehicle_angular_velocity", keys=["xyz"])
        omega_d = np.zeros((1, 3))
        K_W = np.diag([10, 10, 1.5])
        omega_err = np.linalg.norm(K_W @ (omega - omega_d).T, axis=0)
        return t, omega_err
    
    def read_vector_from_dataset(self, dataset_name, keys, num_entry=3):
        t = self.ulog.get_dataset(dataset_name).data["timestamp"]
        data_len = len(t)
        print(f"data length of {dataset_name}: {data_len}")
        data = np.zeros((data_len, num_entry))
        
        for i in range(num_entry):
            if (len(keys) != num_entry):
                data[:, i] = self.ulog.get_dataset(dataset_name).data["{}[{}]".format(keys[0], i)]
            else:
                data[:, i] = self.ulog.get_dataset(dataset_name).data[keys[i]]

        print(f"input resampled data length of {dataset_name}: {len(t)}")
        t, data = self.data_resample_and_clip(t, data)
        print(f"resampled data length of {dataset_name}: {len(t)}")
        return data, t
    
    """
    q = w, x, y, z 
    input shape: 4 x N
    """
    def _read_attitude_from_dataset(self, dataset_name, type="quat", quat_name="q"):
        data_len = len(self.ulog.get_dataset(dataset_name).data["timestamp"])
        print(f"data length of {dataset_name} in {type}: {data_len}")
        t = self.ulog.get_dataset(dataset_name).data["timestamp"]
        if (type == "quat"):
            q = np.zeros((data_len, 4))
            for i in range(4):
                q[:, i] = self.ulog.get_dataset(dataset_name).data["{}[{}]".format(quat_name, i)]
            t, q = self.data_resample_and_clip(t, q)
            return q, t
        elif (type == "euler"):
            keys = ["roll_body", "pitch_body", "yaw_body"]
            euler, t = self.read_vector_from_dataset(dataset_name, keys)
            return euler, t

    """
    1. Resample the data to the reference timestamp, and 
    2. Clip the data to the time interval [t1, t2]
    """
    def data_resample_and_clip(self, t, d):
        ts, ds = FlightDataVisualizer._resampling(self.ref_timestamp, [], t, d)
        print(f"middle resampled data length: {len(ts)}")
        ts, ds = FlightDataVisualizer._time_vector_clipping(self.armed_time, ts, ds, self.t1, self.t2, size=self.len_clipped_timestamp)
        return ts, ds

    """
    Clip the data to the time interval [t1, t2]
    @param t0: the initial time stamp of the dataset (in microsecond)
    @param t: the time stamp of the dataset (in microsecond)
    @param d: the data of the dataset, the shape is N x M, where N is the number of data points and M is the dimension of the data
    @param t1: the start time of the interval (in second)
    @param t2: the end time of the interval (in second)
    """
    @staticmethod
    def _time_vector_clipping(t0, t, d, t1, t2, size=None):
        t1 = t1 * 1e6 + t0
        t2 = t2 * 1e6 + t0
        idx1 = FlightDataVisualizer._fast_nearest_interp(t1, t, np.arange(len(t)))
        if size != None:
            idx2 = t1 + size
        else:
            idx2 = FlightDataVisualizer._fast_nearest_interp(t2, t, np.arange(len(t)))
        if (idx2 > len(t)):
            idx2 = None
        return (t[idx1:idx2] - t0) / 1e6, d[idx1:idx2]
    
    """
    Resample the data (t2, td) to the same time stamp as t1
    """
    @staticmethod
    def _resampling(t1, d1, t2, d2):
        idx = FlightDataVisualizer._fast_nearest_interp(t1, t2, np.arange(len(t2)))
        violation = idx >= len(t2)
        idx[idx >= len(t2)] = len(t2) -1
        ts = t2[idx]
        ds = d2[idx]
        ds[violation, :] = np.NaN
        return ts, ds

    """
    The input quaternion should be in the shape of 
    [[w1, x1, y1, z1],
     [w2, x2, y2, z2],
     ...
     [wn, xn, yn, zn]]
    """
    @staticmethod
    def _trans_wxyz_to_xyzw_batch(q):
        return np.roll(q, -1, axis=1)

    @staticmethod
    def _fast_nearest_interp(xi, x, y):
        """Assumes that x is monotonically increasing!!."""
        # Shift x points to centers
        spacing = np.diff(x) / 2
        x = x + np.hstack([spacing, spacing[-1]])
        # Append the last point in y twice for ease of use
        y = np.hstack([y, y[-1]])
        return y[np.searchsorted(x, xi)]

def plot_vector3f(t, data, keys, title, colors=["#0072BD", "#D95319", "#EDB120"], limits=None):
    gs = gridspec.GridSpec(3,1)
    fig = plt.figure(figsize=(5, 4))
    ax = [0, 0, 0]

    for i in range(3):
        ax[i] = fig.add_subplot(gs[i], sharex= ax[0] if i != 0 else None)
        ax[i].plot(t, data[:, i], color=colors[i])
        ax[i].set_ylabel(r'${}$'.format(keys[i]), fontsize=16)
        if i != 2:
            plt.setp(ax[i].get_xticklabels(), visible=False)
        if limits is not None:
            ax[i].set_ylim(limits[i])

    ax[i].set_xlabel(r'$t$', fontsize=16)

    fig.align_ylabels()
    # plt.suptitle(title, fontsize=16)
    plt.savefig("{}.eps".format(title), format="eps", bbox_inches="tight")
    plt.savefig("{}.svg".format(title), format="svg", bbox_inches="tight")
    plt.savefig("{}.pdf".format(title), format="pdf", bbox_inches="tight")
    plt.show()

"""
Data has the shape of M x [N x 3], where M is the number of data points, N is the number of data sets, and 3 is the dimension of the data
"""
def plot_vector3f_batch(t, data, keys, title, limits=None, marker_style=None, legend=None):
    colors = ["#0072BD", "#D95319", "#EDB120"]
    gs = gridspec.GridSpec(3,1)
    fig = plt.figure(figsize=(5, 4))
    ax = [0, 0, 0]

    num_instance = len(data)
    num_entry = data[0].shape[1]   # x, y, z

    for i in range(num_entry):    # x, y, z
        ax[i] = fig.add_subplot(gs[i], sharex= ax[0] if i != 0 else None)
        for j in range(num_instance):
            ax[i].plot(t, data[j][:, i], color=colors[i], linestyle="-" if marker_style is None else marker_style[j], linewidth=2)
        ax[i].set_ylabel(r'${}$'.format(keys[i]), fontsize=16)
        if i != 2:
            plt.setp(ax[i].get_xticklabels(), visible=False)
        if limits is not None:
            ax[i].set_ylim(limits[i])
        if legend is not None:
            ax[i].legend(legend, loc="upper right", ncol=num_entry)


    ax[i].set_xlabel(r'$t$', fontsize=16)

    fig.align_ylabels()
    # plt.suptitle(title, fontsize=16)
    plt.savefig("{}.eps".format(title), format="eps", bbox_inches="tight")
    plt.savefig("{}.svg".format(title), format="svg", bbox_inches="tight")
    plt.savefig("{}.pdf".format(title), format="pdf", bbox_inches="tight")
    plt.show()

def plot_vector3f_batch_separate(t, data, keys, title, limits=None, legend=None, colors=["#0072BD", "#D95319", "#EDB120"]):
    
    num_instance = len(data)

    gs = gridspec.GridSpec(num_instance,1)
    fig = plt.figure(figsize=(5, 4))
    ax = [ 0 for i in range(num_instance)]


    for i in range(num_instance):    # x, y, z
        ax[i] = fig.add_subplot(gs[i], sharex= ax[0] if i != 0 else None)
        num_entry = data[i].shape[1]
        if colors is None and num_entry == 3:
            colors=["#0072BD", "#D95319", "#EDB120"]
        else:
            colormap = cm.get_cmap('jet', num_entry+1)
            colors = colormap(np.arange(num_entry))

        for j in range(num_entry):
            ax[i].plot(t, data[i][:, j], color=colors[j], linestyle="-", linewidth=2)
        ax[i].set_ylabel(r'${}$'.format(keys[i]), fontsize=16)
        if i != 2:
            plt.setp(ax[i].get_xticklabels(), visible=False)
        if limits is not None:
            ax[i].set_ylim(limits[i])
        if legend is not None:
            ax[i].legend(legend, loc="upper right", ncol=num_entry)


    ax[i].set_xlabel(r'$t$', fontsize=16)

    fig.align_ylabels()
    # plt.suptitle(title, fontsize=16)
    plt.savefig("{}.eps".format(title), format="eps", bbox_inches="tight")
    plt.savefig("{}.svg".format(title), format="svg", bbox_inches="tight")
    plt.savefig("{}.pdf".format(title), format="pdf", bbox_inches="tight")
    plt.show()

def plot_vector3f_with_desire(t, data, t_d, data_d, keys, title, limits=None):
    colors = ["#0072BD", "#D95319", "#EDB120"]
    gs = gridspec.GridSpec(3,1)
    fig = plt.figure(figsize=(5, 4))
    ax = [0, 0, 0]

    for i in range(3):
        if i == 0:
            ax[i] = fig.add_subplot(gs[i])
        else:
            ax[i] = fig.add_subplot(gs[i], sharex=ax[0])

        ax[i].plot(t, data[:, i], color=colors[i], linestyle="-", linewidth=2)
        ax[i].plot(t_d, data_d[:, i], color=colors[i], linestyle="--", linewidth=2)
        ax[i].set_ylabel(r'${}$'.format(keys[i]), fontsize=16)
        if limits is not None:
            ax[i].set_ylim(limits[i])
        if i != 2:
            plt.setp(ax[i].get_xticklabels(), visible=False)

    ax[2].set_xlabel(r'$t$', fontsize=16)

    fig.align_ylabels()
    # plt.suptitle(title, fontsize=16)
    plt.savefig("{}.eps".format(title), format="eps", bbox_inches="tight")
    plt.savefig("{}.svg".format(title), format="svg", bbox_inches="tight")
    plt.savefig("{}.pdf".format(title), format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()