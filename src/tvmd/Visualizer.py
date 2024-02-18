from collections import defaultdict
from pyulog import ULog

import os
import tf
import numpy as np

class FlightDataVisualizer:
    def __init__(self, fname, outfolder, t1, t2) -> None:
        # Extract output folder 
        if outfolder == "default":
            FlightDataVisualizer.outfolder = os.path.join(os.path.dirname(fname), "figures")
        else:
            FlightDataVisualizer.outfolder = outfolder
        print("output folder: ", FlightDataVisualizer.outfolder)
        if not os.path.exists(FlightDataVisualizer.outfolder):
            print("The output folder does not exist, create one.")
            os.makedirs(FlightDataVisualizer.outfolder)

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
        self.t, self.ref_timestamp = FlightDataVisualizer._time_vector_clipping(self.armed_time, self.ref_timestamp, self.ref_timestamp, t1, t2)
        self.len_clipped_timestamp = len(self.ref_timestamp)
        print("ref_timestamp: ", self.len_clipped_timestamp)
        
        FlightDataVisualizer.x_cut_lines = []

    def set_gains(self, K_x, K_v, K_R, K_W):
        self.Kx = np.diag(K_x)
        self.Kv = np.diag(K_v)
        self.K_R = np.diag(K_R)
        self.K_W = np.diag(K_W)

    def read_quaternion_attitude(self, dataset_name):
        q, t = self._read_attitude_from_dataset(dataset_name, type="quat", quat_name="q")
        q_reset, t = self._read_attitude_from_dataset(dataset_name, type="quat", quat_name="delta_q_reset")
        q_reset = FlightDataVisualizer._trans_wxyz_to_xyzw_batch(q_reset)
        q = FlightDataVisualizer._trans_wxyz_to_xyzw_batch(q)
        q_correct = np.zeros((len(t), 4))
        for i in range(len(t)):
            # q_correct[i, :] = tf.transformations.quaternion_multiply(q[i, :], q_reset[i, :])
            q_correct[i, :] = q[i, :]
        return q_correct, t

    def extract_attitude_error_info(self):
        euler_d, t_d = self._read_attitude_from_dataset("vehicle_attitude_setpoint", type="euler")
        # q, t = self._read_attitude_from_dataset("vehicle_attitude", type="quat", quat_name="q")
        # q_reset, t = self._read_attitude_from_dataset("vehicle_attitude", type="quat", quat_name="delta_q_reset")
        # q_reset = FlightDataVisualizer._trans_wxyz_to_xyzw_batch(q_reset)
        # q = FlightDataVisualizer._trans_wxyz_to_xyzw_batch(q)
        q, t = self.read_quaternion_attitude("vehicle_attitude")

        psi = np.zeros(len(t))
        for i in range(len(t)):
            R_d = tf.transformations.euler_matrix(euler_d[i, 0], euler_d[i, 1], euler_d[i, 2], 'sxyz')
            # q_correct = tf.transformations.quaternion_multiply(q[i, :], q_reset[i, :])
            # print("q: ", q[i, :])
            # print("q_correct: ", q_correct)
            # q_correct = tf.transformations.quaternion_multiply(q_reset[i, :], q[i, :])
            # q_correct[0] = -q_correct[0]
            R = tf.transformations.quaternion_matrix(q[i, :])
            psi[i] = 0.5 * np.trace(self.K_R @ (np.eye(4) - R_d.T @ R))
        return t, psi
    
    def extract_position_error_info(self, pos, pos_d, vel, vel_d):
        e_x = (pos - pos_d)
        e_v = (vel - vel_d)
        err_x = np.linalg.norm(e_x @ self.Kx, axis=1)
        err_v = np.linalg.norm(e_v @ self.Kv, axis=1)
        err_sum = err_x + err_v
        return e_x, e_v, err_x, err_v, err_sum

    def extract_angular_velocity_error_info(self):
        omega, t = self.read_vector_from_dataset("vehicle_angular_velocity", keys=["xyz"])
        omega_d = np.zeros((1, 3))
        omega_err = np.linalg.norm(self.K_W @ (omega - omega_d).T, axis=0)
        return t, omega_err
    
    def extract_attitude_error_info(self, q, q_d, w, w_d):
        def vee(so3):
            return np.array([so3[2, 1], so3[0, 2], so3[1, 0]])
        e_w = (w - w_d)
        err_w = np.linalg.norm(self.K_W @ e_w.T, axis=0)
        
        e_R = np.zeros([q_d.shape[0], 3])
        psi = np.zeros(q_d.shape[0])
        for i in range(q_d.shape[0]):
            if q_d.shape[1] == 3:
                # Euler angles
                R_d = tf.transformations.euler_matrix(q_d[i, 0], q_d[i, 1], q_d[i, 2], 'sxyz')
            elif q_d.shape[1] == 4:
                R_d = tf.transformations.quaternion_matrix(q_d[i, :])
            else:
                raise ValueError("The shape of q_d is not correct.")
            
            R = tf.transformations.quaternion_matrix(q[i, :])
            psi[i] = 0.5 * np.trace(self.K_R @ (np.eye(4) - R_d.T @ R))
            e_R[i, :] = 0.5 * vee(R_d.T @ R - R.T @ R_d)

        return e_R, e_w, psi, err_w
        
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
        # print(f"middle resampled data length: {len(ts)}")
        # ts, ds = FlightDataVisualizer._time_vector_clipping(self.armed_time, ts, ds, self.t1, self.t2, size=self.len_clipped_timestamp)
        return self.t, ds

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
