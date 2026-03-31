import numpy as np


def maybe_wrap_innovation(h_system, residual, sensor_axis):
    if hasattr(h_system, "wrap_innovation"):
        return h_system.wrap_innovation(residual, sensor_axis=sensor_axis)
    return residual


def edge_kalman_gain(h_system, r_array, j_matrix, x_pred, measurements):
    H_T = h_system.jacobian(x_pred, 1)[:, 0, ...]
    r_inv = 1 / (r_array ** 2)
    measurement_residual = measurements[None, ..., None] - h_system(x_pred)[..., None]
    # Residual shape is (source_node, sensor_node, measurement_dim, 1) in this path.
    measurement_residual = maybe_wrap_innovation(h_system, measurement_residual, sensor_axis=1)
    y_diff = H_T @ r_inv[None, :, None, None] @ measurement_residual
    y_local_delta_unnorm = (j_matrix[..., None, None] * y_diff).sum(1)
    y_local_delta = y_local_delta_unnorm / j_matrix.sum(0)[..., None, None]
    return y_local_delta


def centralized_extended_kalman_filter(measurements, f_system, h_system, r_array, q, p0, x0,
                                       time_steps, node_num):
    state_dim = x0.shape[0]  # Infer state dimension from initial state
    x_hat = np.zeros((time_steps, state_dim, 1))
    measurements = measurements.transpose(2, 0, 1)
    r_inv = 1 / (r_array ** 2)
    p = p0
    x_pred = x0
    for i in range(measurements.shape[0]):
        H_T = h_system.jacobian(x_pred[None, ...], 1)[0, 0, ...]
        H = H_T.transpose((0, 2, 1))
        node_s = H_T @ r_inv[:, None, None] @ H
        measurement_residual = measurements[i, ..., None] - h_system(x_pred[None, ...], 1)
        measurement_residual = maybe_wrap_innovation(h_system, measurement_residual, sensor_axis=0)
        y_node_delta = H_T @ r_inv[:, None, None] @ measurement_residual
        s_all = np.sum(node_s, axis=0) / node_num
        y_all_delta = np.sum(y_node_delta, axis=0) / node_num
        M = np.linalg.inv(np.linalg.inv(p) + s_all)
        x_current = x_pred + M @ y_all_delta
        F = f_system.jacobian(x_current[None, ..., 0])[0, ...]
        p = F @ M @ F.T + node_num * q ** 2 * np.eye(state_dim)
        x_pred = f_system(x_current)
        x_hat[i, ...] = x_current
    return x_hat


def diffusion_extended_kalman_filter(measurements, f_system, h_system, r_array, q, p0, x0, j_matrix,
                                     time_steps, node_num):
    state_dim = x0.shape[0]  # Infer state dimension from initial state
    x_hat = np.zeros((time_steps, node_num, state_dim, 1))
    r_inv = 1 / (r_array ** 2)
    measurements = measurements.transpose(2, 0, 1)
    p = p0[None, ...].repeat(node_num, axis=0)
    x_pred = x0
    for i in range(measurements.shape[0]):
        x_current_list = []
        m_list = []
        if i == 0:
            x_pred = x_pred[None, ...].repeat(node_num, axis=0)
        H_T = h_system.jacobian(x_pred, 1)[:, 0, ...]
        H = H_T.transpose((0, 1, 3, 2))
        for j in range(node_num):
            x_pred_j = x_pred[j, ...][None, ...]
            H_T_j = H_T[j, ...]
            H_j = H[j, ...]
            node_s = H_T_j @ r_inv[:, None, None] @ H_j
            measurement_residual_j = measurements[i, ..., None] - h_system.func(x_pred_j, 1)
            measurement_residual_j = maybe_wrap_innovation(h_system, measurement_residual_j, sensor_axis=0)
            y_node_delta_j = H_T_j @ r_inv[:, None, None] @ measurement_residual_j
            s_local_j = (j_matrix[:, j, None, None] * node_s).sum(0) / j_matrix[:, j].sum()
            y_local_delta_j = (j_matrix[:, j, None, None] * y_node_delta_j).sum(0) / j_matrix[:, j].sum()
            M_j = np.linalg.inv(np.linalg.inv(p[j, ...]) + s_local_j)
            x_current_j = (x_pred_j + M_j @ y_local_delta_j)[0, ...]
            x_current_list.append(x_current_j)
            m_list.append(M_j)
        x_current_all = np.stack(x_current_list, axis=0)
        M_all = np.stack(m_list, axis=0)
        x_current = np.einsum('ij, jmk -> imk', j_matrix, x_current_all) / j_matrix.sum(axis=0)[..., None, None]
        F = f_system.jacobian(x_current[..., 0])
        p = F @ M_all @ F.transpose((0, 2, 1)) + j_matrix.sum(axis=0)[..., None, None] * q ** 2 * np.eye(state_dim)
        x_pred = f_system(x_current)
        x_hat[i, ...] = x_current
    return x_hat


def diffusion_extended_kalman_filter_parallel_edge(measurements, f_system, h_system, r_array, q, p0, x0, j_matrix,
                                                   time_steps, node_num):
    state_dim = x0.shape[0]  # Infer state dimension from initial state
    x_hat = np.zeros((time_steps, node_num, state_dim, 1))
    r_inv = 1 / (r_array ** 2)
    measurements = measurements.transpose(2, 0, 1)
    p = p0[None, ...].repeat(node_num, axis=0)
    x_pred = x0
    for i in range(measurements.shape[0]):
        x_current_list = []
        m_list = []
        if i == 0:
            x_pred = x_pred[None, ...].repeat(node_num, axis=0)
        H_T = h_system.jacobian(x_pred, 1)[:, 0, ...]
        H = H_T.transpose((0, 1, 3, 2))
        y_kalman_edge = edge_kalman_gain(h_system, r_array, j_matrix, x_pred, measurements[i, ...])
        for j in range(node_num):
            x_pred_j = x_pred[j, ...][None, ...]
            H_T_j = H_T[j, ...]
            H_j = H[j, ...]
            node_s = H_T_j @ r_inv[:, None, None] @ H_j
            s_local_j = (j_matrix[:, j, None, None] * node_s).sum(0) / j_matrix[:, j].sum()
            y_local_delta_j = y_kalman_edge[j, ...]
            M_j = np.linalg.inv(np.linalg.inv(p[j, ...]) + s_local_j)
            x_current_j = (x_pred_j + M_j @ y_local_delta_j)[0, ...]
            x_current_list.append(x_current_j)
            m_list.append(M_j)
        x_current_all = np.stack(x_current_list, axis=0)
        M_all = np.stack(m_list, axis=0)
        x_current = np.einsum('ij, jmk -> imk', j_matrix, x_current_all) / j_matrix.sum(axis=0)[..., None, None]
        F = f_system.jacobian(x_current[..., 0])
        p = F @ M_all @ F.transpose((0, 2, 1)) + j_matrix.sum(axis=0)[..., None, None] * q ** 2 * np.eye(state_dim)
        x_pred = f_system(x_current)
        x_hat[i, ...] = x_current
    return x_hat


def local_extended_kalman_filter(measurements, f_system, h_system, r_array, q, p0, x0, j_matrix,
                                 time_steps, node_num):
    state_dim = x0.shape[0]  # Infer state dimension from initial state
    x_hat = np.zeros((time_steps, node_num, state_dim, 1))
    r_inv = 1 / (r_array ** 2)
    measurements = measurements.transpose(2, 0, 1)
    p = p0
    x_pred = x0
    for i in range(measurements.shape[0]):
        x_current_list = []
        m_list = []
        if i == 0:
            x_pred = x_pred[None, ...].repeat(node_num, axis=0)
        h_transpose = h_system.jacobian(x_pred, 1)
        H = h_transpose.transpose((0, 1, 3, 2))
        for j in range(node_num):
            x_pred_j = x_pred[j, ...][None, ...]
            H_T_j = h_transpose[j, ...]
            H_j = H[j, ...]
            node_s = H_T_j @ r_inv[:, None, None] @ H_j
            measurement_residual_j = measurements[i, ..., None] - h_system(x_pred_j, 1)
            measurement_residual_j = maybe_wrap_innovation(h_system, measurement_residual_j, sensor_axis=0)
            y_node_delta_j = H_T_j @ r_inv[:, None, None] @ measurement_residual_j
            s_local_j = (j_matrix[:, j, None, None] * node_s).sum(0) / j_matrix[:, j].sum()
            y_local_delta_j = (j_matrix[:, j, None, None] * y_node_delta_j).sum(0) / j_matrix[:, j].sum()
            M_j = np.linalg.inv(np.linalg.inv(p[j, ...]) + s_local_j)
            x_current_j = (x_pred_j + M_j @ y_local_delta_j)[0, ...]
            x_current_list.append(x_current_j)
            m_list.append(M_j)
        x_current_all = np.stack(x_current_list, axis=0)
        M_all = np.stack(m_list, axis=0)
        F = f_system.jacobian(x_current_all[..., 0])
        p = F @ M_all @ F.transpose((0, 2, 1)) + j_matrix.sum(axis=0)[..., None, None] * q ** 2 * np.eye(state_dim)
        x_pred = f_system(x_current_all)
        x_hat[i, ...] = x_current_all
    return x_hat
