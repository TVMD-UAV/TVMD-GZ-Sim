#!/usr/bin/env python3

'''
This script draws the attitude error function and the angular 
velocity error function from the ULog file.

Author: Yen-Cheng Chu (sciyen.ycc@gmail.com)
'''

from collections import defaultdict
import argparse
from matplotlib import pyplot as plt

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

    viz = FlightDataVisualizer(args.filename)
    t, psi = viz.extract_attitude_error_info()
    t_omega, omega_err = viz.extract_angular_velocity_error_info()
    t_omega_s, omega_err_s = FlightDataVisualizer.resampling(t, psi, t_omega, omega_err[np.newaxis, :])
    att_err_sum = psi + omega_err_s[0, :]


    start_time = 49.5
    end_time = 60.5
    start_time = 5
    end_time = 40
    t, psi = viz.time_vector_resampling(t, psi, start_time, end_time)
    t_omega, omega_err = viz.time_vector_resampling(t_omega, omega_err, start_time, end_time)
    t_sum, att_err_sum = viz.time_vector_resampling(t_omega_s, att_err_sum, start_time, end_time)
    
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, psi)
    plt.xlabel(r'$t$', fontsize=16)
    plt.ylabel(r'$\Psi(\mathbf{R}, \mathbf{R}_d)$', fontsize=16)

    plt.subplot(3, 1, 2)
    plt.plot(t_omega, omega_err)
    plt.xlabel(r'$t$', fontsize=16)
    plt.ylabel(r'$\|\mathbf{e}_{\Omega}\|_2$', fontsize=16)

    plt.subplot(3, 1, 3)
    plt.plot(t_sum, att_err_sum)
    plt.xlabel(r'$t$', fontsize=16)
    plt.ylabel(r'$\Psi + \|\mathbf{e}_{\Omega}\|_2$', fontsize=16)

    plt.show()


class FlightDataVisualizer:
    def __init__(self, fname) -> None:
        self.ulog = ULog(fname)
        self.data = self.ulog.data_list
        self.multiids = defaultdict(set)
        for d in self.data:
            self.multiids[d.name].add(d.multi_id)
            
        # print(data)
        # print(len(data))
        # print(multiids)

    def extract_attitude_error_info(self):
        euler_d, t_d = self._read_attitude_from_dataset("vehicle_attitude_setpoint", type="euler")
        q, t = self._read_attitude_from_dataset("vehicle_attitude", type="quat", quat_name="q")
        q_reset, t = self._read_attitude_from_dataset("vehicle_attitude", type="quat", quat_name="delta_q_reset")
        q_reset = FlightDataVisualizer._trans_wxyz_to_xyzw_batch(q_reset)
        q = FlightDataVisualizer._trans_wxyz_to_xyzw_batch(q)

        psi = np.zeros(len(t))
        t_ds, euler_ds = FlightDataVisualizer.resampling(t, q, t_d, euler_d)

        for i in range(len(t)):
            R_d = tf.transformations.euler_matrix(euler_ds[0, i], euler_ds[1, i], euler_ds[2, i], 'sxyz')
            q_correct = tf.transformations.quaternion_multiply(q[:, i], q_reset[:, i])
            # print("q: ", q[:, i])
            # print("q_correct: ", q_correct)
            # q_correct = tf.transformations.quaternion_multiply(q_reset[:, i], q[:, i])
            # q_correct[0] = -q_correct[0]
            R = tf.transformations.quaternion_matrix(q_correct)
            # print(R)
            # print(R_d)
            K_R = np.diag([5, 5, 0, 0])
            psi[i] = 0.5 * np.trace(K_R @ (np.eye(4) - R_d.T @ R))
            print(psi)
        return t, psi
    
    def extract_angular_velocity_error_info(self):
        omega, t = self._read_vector3f_from_dataset("vehicle_angular_velocity", keys=["xyz"])
        omega_d = np.zeros((3, 1))
        K_W = np.diag([5, 5, 0])
        omega_err = np.linalg.norm(K_W @ (omega - omega_d), axis=0)
        return t, omega_err
        
    def time_vector_resampling(self, t, d, t1, t2):
        t0 = t[0]
        t1 = t1 * 1e6 + t0
        t2 = t2 * 1e6 + t0
        idx1 = FlightDataVisualizer.fast_nearest_interp(t1, t, np.arange(len(t)))
        idx2 = FlightDataVisualizer.fast_nearest_interp(t2, t, np.arange(len(t)))
        return (t[idx1:idx2] - t0) / 1e6, d[idx1:idx2]
    
    def _read_vector3f_from_dataset(self, dataset_name, keys):
        data_len = len(self.ulog.get_dataset(dataset_name).data["timestamp"])
        print("data length of {}: {}".format(dataset_name, data_len))
        t = self.ulog.get_dataset(dataset_name).data["timestamp"]
        data = np.zeros((3, data_len))
        
        for i in range(3):
            if (len(keys) != 3):
                data[i, :] = self.ulog.get_dataset(dataset_name).data["{}[{}]".format(keys[0], i)]
            else:
                data[i, :] = self.ulog.get_dataset(dataset_name).data[keys[i]]
        return data, t
    """
    q = w, x, y, z 
    input shape: 4 x N
    """
    def _read_attitude_from_dataset(self, dataset_name, type="quat", quat_name="q"):
        data_len = len(self.ulog.get_dataset(dataset_name).data["timestamp"])
        print("data length of attitude: {}".format(data_len))
        t = self.ulog.get_dataset(dataset_name).data["timestamp"]
        if (type == "quat"):
            q = np.zeros((4, data_len))
            for i in range(4):
                q[i, :] = self.ulog.get_dataset(dataset_name).data["{}[{}]".format(quat_name, i)]
            return q, t
        elif (type == "euler"):
            keys = ["roll_body", "pitch_body", "yaw_body"]
            # euler = np.zeros((3, data_len))
            # for i in range(3):
            #     euler[i, :] = self.ulog.get_dataset(dataset_name).data[keys[i]]
            euler, t = self._read_vector3f_from_dataset(dataset_name, keys)
            return euler, t

    """
    Resample the data to the same time stamp
    """
    @staticmethod
    def resampling(t1, d1, t2, d2):
        idx = FlightDataVisualizer.fast_nearest_interp(t1, t2, np.arange(len(t2)))
        idx[idx >= len(t2)] = len(t2) - 1
        ts = t2[idx]
        ds = d2[:, idx]
        return ts, ds

    @staticmethod
    def _trans_wxyz_to_xyzw_batch(q):
        return np.roll(q, -1, axis=0)

    @staticmethod
    def fast_nearest_interp(xi, x, y):
        """Assumes that x is monotonically increasing!!."""
        # Shift x points to centers
        spacing = np.diff(x) / 2
        x = x + np.hstack([spacing, spacing[-1]])
        # Append the last point in y twice for ease of use
        y = np.hstack([y, y[-1]])
        return y[np.searchsorted(x, xi)]

if __name__ == '__main__':
    main()