import numpy as np
import tensorflow as tf

# apply transform to points
# - transform pointcloud into given frame

class SO3:

    DoF = 3

    epsilon = 1e-10

    @staticmethod
    def eyes_like(reference):

        # Parse reference tensor.
        assert reference.shape[-1] == 4
        shape = tf.shape(reference)
        dtype = reference.dtype

        # Build matching identity tensor.
        identity = [3 * [0.0] + [1.0]]
        identity = tf.convert_to_tensor(identity, dtype)
        return tf.broadcast_to(identity, shape)

    @staticmethod
    def inverse(Rab):

        # Convert input to tensor.
        Rab = tf.convert_to_tensor(Rab)

        # Check if correct dimensions.
        assert Rab.shape[-1] == 4

        # Compute squared norm of quaternion vector.
        squared_norm = tf.reduce_sum(tf.square(Rab), -1, True)

        # Compute inverse of quaternion.
        return SO3._conjugate(Rab) / squared_norm

    @staticmethod
    def rotate_rotation(Rab, Rbc):

        # Convert input to tensors.
        Rab = tf.convert_to_tensor(Rab)
        Rbc = tf.convert_to_tensor(Rbc)

        # Compute as if multiple rotations.
        Rbc = tf.expand_dims(Rbc, axis=-2)
        Rac = SO3.rotate_rotations(Rab, Rbc)
        return tf.squeeze(Rac, axis=-2)

    @staticmethod
    def rotate_rotations(Rab, Rbc):

        # Convert input to tensors.
        Rab = tf.convert_to_tensor(Rab)
        Rbc = tf.convert_to_tensor(Rbc)

        # Check if correct dimensions.
        assert Rab.shape[-1] == 4
        assert Rbc.shape[-1] == 4

        # Check if shapes match.
        assert Rab.shape.as_list()[:-1] == Rbc.shape.as_list()[:-2]

        # Expand dimensions accordingly.
        Rab = tf.expand_dims(Rab, axis=-2)

        # Get components of quaternions.
        x1, y1, z1, w1 = tf.unstack(Rab, axis=-1)
        x2, y2, z2, w2 = tf.unstack(Rbc, axis=-1)

        # Compute components of new quaternion.
        x =  x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
        y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
        z =  x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
        w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2

        # Combine components into quaternion.
        return tf.stack((x, y, z, w), axis=-1)

    @staticmethod
    def rotate_point(Rab, Xb):

        # Convert input to tensors.
        Rab = tf.convert_to_tensor(Rab)
        Xb = tf.convert_to_tensor(Xb)

        # Compute as if multiple rotations.
        Xb = tf.expand_dims(Xb, axis=-2)
        Xa = SO3.rotate_points(Rab, Xb)
        return tf.squeeze(Xa, axis=-2)

    @staticmethod
    def rotate_points(Rab, Xb):

        # Convert input to tensors.
        Rab = tf.convert_to_tensor(Rab)
        Xb = tf.convert_to_tensor(Xb)

        # Check if correct dimensions.
        assert Rab.shape[-1] == 4
        assert Xb.shape[-1] == 3

        # Check if shapes match.
        assert Rab.shape.as_list()[:-1] == Xb.shape.as_list()[:-2]

        # Create padding to convert 3D points to 4D vectors.
        padding = [[0, 0] for _ in range(Xb.shape.ndims - 1)] + [[0, 1]]

        # Zero-pad 3D points to 4D vectors.
        Xb = tf.pad(Xb, paddings=padding)

        # Apply left-hand-side of rotation.
        Xab = SO3.rotate_rotations(Rab, Xb)

        # Expand dimensions accordingly.
        Rab = tf.expand_dims(Rab, axis=-2)

        # Compute inverse rotation.
        Rba = SO3._conjugate(Rab)

        # Apply right-hand-side of rotation.
        return SO3._imaginary_rotate_rotation(Xab, Rba)

    @staticmethod
    def log(quat):

        # Return tangent value only.
        return SO3.log_theta(quat)[0]

    @staticmethod
    def log_theta(quat):

        # Convert input to tensors.
        quat = tf.convert_to_tensor(quat)

        # Check if correct dimensions.
        assert quat.shape[-1] == 4

        # Split quaternion into real and imaginary parts.
        xyz, w = tf.split(quat, (3, 1), -1)

        # Compute squared norm of imaginary vector.
        squared_n = tf.reduce_sum(tf.square(xyz), -1, keepdims=True)

        # Compute norm of imaginary vector.
        n = tf.sqrt(squared_n)
        small_n = n < SO3.epsilon
        n = tf.where(small_n, tf.ones_like(n), n)

        # Recompute real part to avoid divide-by-zero.
        large_abs_w = tf.abs(w) >= SO3.epsilon
        w = tf.where(large_abs_w, w, tf.ones_like(w))
        squared_w = w * w

        # Check if norm of imaginary part near zero.
        two_atan_nbyw_by_n = tf.where(small_n,
            2 / w - 2 * squared_n / (w * squared_w),
            # Check if large magnitude of real part.
            tf.where(large_abs_w,
                2 * tf.atan(n / w) / n,
                # Check if positive real part.
                tf.where(w > 0.0, +np.pi / n, -np.pi / n)))

        # Compute final tangent and theta.
        tangent = two_atan_nbyw_by_n * xyz
        theta = two_atan_nbyw_by_n * n
        return tangent, theta

    @staticmethod
    def exp(omega):

        # Return quaternion value only.
        return SO3.exp_theta(omega)[0]

    @staticmethod
    def exp_theta(omega):

        # Convert input to tensor.
        omega = tf.convert_to_tensor(omega)

        # Check if correct dimensions.
        assert omega.shape[-1] == 3

        # Compute theta^2 and theta^4.
        theta_sq = tf.reduce_sum(tf.square(omega), -1, keepdims=True)
        theta_pow4 = theta_sq * theta_sq

        # Compute theta and half-theta.
        theta = tf.sqrt(theta_sq)
        half_theta = 0.5 * theta
        small_theta = theta < SO3.epsilon
        theta = tf.where(small_theta, tf.ones_like(theta), theta)

        # Compute real factor.
        real_factor = tf.where(small_theta,
            1 - 1 / 8 * theta_sq + 1 / 384 * theta_pow4,
            tf.cos(half_theta))

        # Compute imaginary factor.
        imag_factor = tf.where(small_theta,
            0.5 - 1 / 48 * theta_sq + 1 / 3840 * theta_pow4,
            tf.sin(half_theta) / theta)

        # Compute imaginary vector.
        imag_vector = imag_factor * omega

        # Build final unit quaternion for SO3.
        quat = tf.concat((imag_vector, real_factor), -1)

        # Return quaternion and theta.
        return quat, theta

    @staticmethod
    def hat(omega):

        # Convert input to tensor.
        omega = tf.convert_to_tensor(omega)

        # Check if correct dimensions.
        assert omega.shape[-1] == 3

        # Get components of omega.
        ox, oy, oz = tf.unstack(omega, axis=-1)

        # Create matching tensor of zeros.
        _0 = tf.zeros_like(ox)

        # Create individual rows of output.
        row0 = tf.stack([  _0, -oz,  oy ], -1)
        row1 = tf.stack([  oz,  _0, -ox ], -1)
        row2 = tf.stack([ -oy,  ox,  _0 ], -1)

        # Return final tensor of 3x3 matrices.
        return tf.stack([ row0, row1, row2 ], -2)

    @staticmethod
    def _conjugate(Rab):

        # Convert input to tensor.
        Rab = tf.convert_to_tensor(Rab)

        # Check if correct dimensions.
        assert Rab.shape[-1] == 4

        # Split quaternion into real and imaginary parts.
        xyz, w = tf.split(Rab, (3, 1), -1)

        # Compute conjugate of quaternion.
        return tf.concat((-xyz, w), -1)

    @staticmethod
    def _imaginary_rotate_rotation(Rab, Rbc):

        # Compute as if multiple rotations.
        Rbc = tf.expand_dims(Rbc, axis=-2)
        Rac = SO3._imaginary_rotate_rotations(Rab, Rbc)
        return tf.squeeze(Rac, axis=-2)

    @staticmethod
    def _imaginary_rotate_rotations(Rab, Rbc):

        # Expand dimensions accordingly.
        Rab = tf.expand_dims(Rab, axis=-2)

        # Get components of quaternions.
        x1, y1, z1, w1 = tf.unstack(Rab, axis=-1)
        x2, y2, z2, w2 = tf.unstack(Rbc, axis=-1)

        # Compute imaginary part of new quaternion.
        x =  x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
        y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
        z =  x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2

        # Combine imaginary part into vector.
        return tf.stack((x, y, z), axis=-1)

class SE3:

    DoF = 6

    @staticmethod
    def eyes_like(reference):

        # Parse reference tensor.
        assert reference.shape[-1] == 7
        shape = tf.shape(reference)
        dtype = reference.dtype

        # Build matching identity tensor.
        identity = [6 * [0.0] + [1.0]]
        identity = tf.convert_to_tensor(identity, dtype)
        return tf.broadcast_to(identity, shape)

    @staticmethod
    def inverse(Tab):

        # Convert input to tensor.
        Tab = tf.convert_to_tensor(Tab)

        # Check if correct dimensions.
        assert Tab.shape[-1] == 7

        # Get translation.
        tab = Tab[..., 0:3]

        # Get quaternion.
        Rab = Tab[..., 3:7]

        # Invert rotation.
        Rba = SO3.inverse(Rab)

        # Invert translation.
        tba = -SO3.rotate_point(Rba, tab)

        # Recombine components.
        return tf.concat((tba, Rba), -1)

    @staticmethod
    def transform_transform(Tab, Tbc):

        # Convert input to tensors.
        Tab = tf.convert_to_tensor(Tab)
        Tbc = tf.convert_to_tensor(Tbc)

        # Compute as if multiple transforms.
        Tbc = tf.expand_dims(Tbc, axis=-2)
        Tac = SE3.transform_transforms(Tab, Tbc)
        return tf.squeeze(Tac, axis=-2)

    @staticmethod
    def transform_transforms(Tab, Tbc):

        # Convert input to tensor.
        Tab = tf.convert_to_tensor(Tab)
        Tbc = tf.convert_to_tensor(Tbc)

        # Check if correct dimensions.
        assert Tab.shape[-1] == 7
        assert Tbc.shape[-1] == 7

        # Check if shapes match.
        assert Tab.shape.as_list()[:-1] == Tbc.shape.as_list()[:-2]

        # Get translations.
        tab = Tab[..., 0:3]
        tbc = Tbc[..., 0:3]

        # Get quaternions.
        Rab = Tab[..., 3:7]
        Rbc = Tbc[..., 3:7]

        # Compute new rotation.
        Rac = SO3.rotate_rotations(Rab, Rbc)

        # Compute new translation.
        tac = SO3.rotate_points(Rab, tbc) + tf.expand_dims(tab, -2)

        # Recombine components.
        return tf.concat((tac, Rac), -1)

    @staticmethod
    def transform_point(Tab, Xb):

        # Convert input to tensors.
        Tab = tf.convert_to_tensor(Tab)
        Xb = tf.convert_to_tensor(Xb)

        # Compute as if multiple transforms.
        Xb = tf.expand_dims(Xb, axis=-2)
        Xa = SE3.transform_points(Tab, Xb)
        return tf.squeeze(Xa, axis=-2)

    @staticmethod
    def transform_points(Tab, Xb):

        # Convert input to tensor.
        Tab = tf.convert_to_tensor(Tab)
        Xb = tf.convert_to_tensor(Xb)

        # Check if correct dimensions.
        assert Tab.shape[-1] == 7
        assert Xb.shape[-1] == 3

        # Check if shapes match.
        assert Tab.shape.as_list()[:-1] == Xb.shape.as_list()[:-2]

        # Get quaternion.
        Rab = Tab[..., 3:7]

        # Get translation.
        tab = tf.expand_dims(Tab[..., 0:3], -2)

        # Apply transformation to points.
        return SO3.rotate_points(Rab, Xb) + tab

    @staticmethod
    def log(Tab):

        # Convert input to tensor.
        Tab = tf.convert_to_tensor(Tab)

        # Check if correct dimensions.
        assert Tab.shape[-1] == 7

        # Split into translation and rotation components.
        translation, quat = tf.split(Tab, (3, 4), -1)

        # Compute quaternion and theta.
        omega, theta = SO3.log_theta(quat)

        # Compute twist matrix.
        Omega = SO3.hat(omega)
        Omega_sq = tf.matmul(Omega, Omega)

        # Expand dimensions for batch scaling.
        translation = tf.expand_dims(translation, -1)
        theta = tf.expand_dims(theta, -1)
        half_theta = 0.5 * theta

        # Determine how translation should be computed.
        small_theta = tf.abs(theta) < SO3.epsilon
        theta = tf.where(small_theta, tf.ones_like(theta), theta)
        small_theta = tf.broadcast_to(small_theta, tf.shape(translation))

        # Build identity matrix for batch.
        eye = tf.eye(3, dtype=Omega.dtype)
        identity = tf.broadcast_to(eye, tf.shape(Omega))

        # Compute translation when theta is small.
        small_theta_head = tf.matmul(identity - 0.5 * Omega + 1.0 / 12.0 *
                Omega_sq, translation)

        # Compute translation component when theta is large.
        large_theta_head = tf.matmul(identity - 0.5 * Omega + (1.0 - theta *
                tf.cos(half_theta) / (2.0 * tf.sin(half_theta))) /
                (theta * theta) * Omega_sq, translation)

        # Compute final translation component.
        head = tf.where(small_theta, small_theta_head, large_theta_head)

        # Combine final components.
        head = tf.squeeze(head, -1)
        return tf.concat((head, omega), -1)

    @staticmethod
    def exp(tangent):

        # Convert input to tensor.
        tangent = tf.convert_to_tensor(tangent)

        # Check if correct dimensions.
        assert tangent.shape[-1] == 6

        # Split into translation and rotation components.
        head, omega = tf.split(tangent, (3, 3), -1)

        # Compute quaternion and theta.
        quat, theta = SO3.exp_theta(omega)

        # Compute twist matrix.
        Omega = SO3.hat(omega)
        Omega_sq = tf.matmul(Omega, Omega)

        # Determine how translation should be computed.
        small_theta = theta < SO3.epsilon
        theta = tf.where(small_theta, tf.ones_like(theta), theta)
        small_theta = tf.broadcast_to(small_theta, tf.shape(head))

        # Expand dimensions for batch scaling.
        theta = tf.expand_dims(theta, -1)
        theta_sq = tf.square(theta)

        # Compute translation when theta is small.
        small_theta_translation = SO3.rotate_point(quat, head)

        # Compute translation when theta is large.
        rotation = tf.eye(3, dtype=Omega.dtype)
        rotation = tf.broadcast_to(rotation, tf.shape(Omega))
        rotation += (1.0 - tf.cos(theta)) / (theta_sq) * Omega
        rotation += (theta - tf.sin(theta)) / (theta_sq * theta) * Omega_sq
        large_theta_translation = tf.matmul(rotation, tf.expand_dims(head, -1))
        large_theta_translation = tf.squeeze(large_theta_translation, -1)

        # Compute final translation.
        translation = tf.where(small_theta,
                small_theta_translation, large_theta_translation)

        # Combine components into SE3.
        return tf.concat((translation, quat), -1)

