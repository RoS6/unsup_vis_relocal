import numpy as np
import tensorflow as tf
from transform import SE3
from transform import SO3

class TestSO3:
    
    def test_inverse(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            quat = tf.placeholder(dtype, [ 5, 3, None, 4 ])
            inverse = SO3.inverse(quat)
            assert quat.shape.as_list() == inverse.shape.as_list()

        with tf.Session() as session:

            # Test operation on single quaternion.

            quat = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            expected = [ -0.034270799, 0.10602051, -0.14357218, 0.98334744 ]
            found = SO3.inverse(quat).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple quaternions.

            quat = [[  0.034270799, -0.10602051,  0.14357218, 0.98334744 ],
                    [ -0.255551240,  0.17475706, -0.32758190, 0.89266099 ]]

            expected = [[ -0.034270799,  0.10602051, -0.14357218, 0.98334744 ],
                        [  0.255551240, -0.17475706,  0.32758190, 0.89266099 ]]

            found = SO3.inverse(quat).eval()
            assert np.allclose(expected, found)

    def test_rotate_rotation(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            Rab = tf.placeholder(dtype, [ 5, 3, None, 4 ])
            Rbc = tf.placeholder(dtype, [ 5, 3, None, 4 ])
            Rac = SO3.rotate_rotation(Rab, Rbc)
            assert Rab.shape.as_list() == Rac.shape.as_list()

        with tf.Session() as session:

            # Test operation on single quaternion pair.

            Rab = [  0.034270799, -0.10602051,  0.14357218, 0.98334744 ]
            Rbc = [ -0.255551240,  0.17475706, -0.32758190, 0.89266099 ]
            expected = [ -0.21106331, 0.05174298, -0.21507015, 0.95211332 ]
            found = SO3.rotate_rotation(Rab, Rbc).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple quaternion pairs.

            Rab = [[  0.034270799, -0.10602051,  0.14357218, 0.98334744 ],
                   [  0.034270799, -0.10602051,  0.14357218, 0.98334744 ]]

            Rbc = [[ -0.255551240,  0.17475706, -0.32758190, 0.89266099 ],
                   [ -0.034270799,  0.10602051, -0.14357218, 0.98334744 ]]

            expected = [[ -0.21106331, 0.05174298, -0.21507015, 0.95211332 ],
                        [  0.00000000, 0.00000000,  0.00000000, 1.00000000 ]]

            found = SO3.rotate_rotation(Rab, Rbc).eval()
            assert np.allclose(expected, found)

    def test_rotate_rotations(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            Rab = tf.placeholder(dtype, [ 5, 3, None, 4 ])
            Rbc = tf.placeholder(dtype, [ 5, 3, None, 8, 4 ])
            Rac = SO3.rotate_rotations(Rab, Rbc)
            assert Rbc.shape.as_list() == Rac.shape.as_list()

        with tf.Session() as session:

            # Test operation on single quaternion pair.

            Rab = [  0.034270799, -0.10602051,  0.14357218, 0.98334744 ]

            Rbc = [[ -0.255551240,  0.17475706, -0.32758190, 0.89266099 ],
                   [ -0.034270799,  0.10602051, -0.14357218, 0.98334744 ]]

            expected = [[ -0.21106331, 0.05174298, -0.21507015, 0.95211332 ],
                        [  0.00000000, 0.00000000,  0.00000000, 1.00000000 ]]

            found = SO3.rotate_rotations(Rab, Rbc).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple quaternion pairs.

            Rab = [[  0.034270799, -0.10602051,  0.14357218, 0.98334744 ],
                   [  0.034270799, -0.10602051,  0.14357218, 0.98334744 ]]

            Rbc = [[[ -0.255551240,  0.17475706, -0.32758190, 0.89266099 ],
                    [ -0.034270799,  0.10602051, -0.14357218, 0.98334744 ]],
                   [[ -0.034270799,  0.10602051, -0.14357218, 0.98334744 ],
                    [ -0.255551240,  0.17475706, -0.32758190, 0.89266099 ]]]

            expected = [[[ -0.21106331, 0.05174298, -0.21507015, 0.95211332 ],
                         [  0.00000000, 0.00000000,  0.00000000, 1.00000000 ]],
                        [[  0.00000000, 0.00000000,  0.00000000, 1.00000000 ],
                         [ -0.21106331, 0.05174298, -0.21507015, 0.95211332 ]]]

            found = SO3.rotate_rotations(Rab, Rbc).eval()
            assert np.allclose(expected, found)

    def test_rotate_point(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            quat = tf.placeholder(dtype, [ 5, 3, None, 4 ])
            point = tf.placeholder(dtype, [ 5, 3, None, 3 ])
            found = SO3.rotate_point(quat, point)
            assert point.shape.as_list() == found.shape.as_list()

        with tf.Session() as session:

            # Test operation on single quaternion-point pair.

            Xb = [ -0.1, 0.2, -0.3 ]
            Rab = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            expected = [ -0.091954433, 0.19312845, -0.30699476 ]
            found = SO3.rotate_point(Rab, Xb).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple quaternion-point pairs.

            Xb = [[ -0.1,  0.2, -0.3 ],
                  [  0.4, -0.6,  0.7 ]]

            Rab = [[  0.03427080, -0.10602051,  0.14357218, 0.98334744 ],
                   [ -0.25555124,  0.17475706, -0.32758190, 0.89266099 ]]

            expected = [[ -0.09195443,  0.19312845, -0.30699476 ],
                        [  0.32800570, -0.42330084,  0.85042851 ]]

            found = SO3.rotate_point(Rab, Xb).eval()
            assert np.allclose(expected, found)

    def test_rotate_points(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            quat = tf.placeholder(dtype, [ 5, 3, None, 4 ])
            point = tf.placeholder(dtype, [ 5, 3, None, 8, 3 ])
            found = SO3.rotate_points(quat, point)
            assert point.shape.as_list() == found.shape.as_list()

        with tf.Session() as session:

            # Test operation on single quaternion-point pair.

            Xb = [[ -0.1,  0.2, -0.3 ],
                  [  0.4, -0.6,  0.7 ]]

            Rab = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]

            expected = [[ -0.09195443,  0.19312845, -0.30699476 ],
                        [  0.40922650, -0.53230709,  0.74778529 ]]

            found = SO3.rotate_points(Rab, Xb).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple quaternion-point pairs.

            Xb = [[[ -0.1,  0.2, -0.3 ],
                   [  0.4, -0.6,  0.7 ]],
                  [[ -0.1,  0.2, -0.3 ],
                   [  0.4, -0.6,  0.7 ]]]

            Rab = [[  0.03427080, -0.10602051,  0.14357218, 0.98334744 ],
                   [ -0.25555124,  0.17475706, -0.32758190, 0.89266099 ]]

            expected = [[[ -0.09195443,  0.19312845, -0.30699476 ],
                         [  0.40922650, -0.53230709,  0.74778529 ]],
                        [[ -0.11715360,  0.09584523, -0.34218230 ],
                         [  0.32800570, -0.42330084,  0.85042851 ]]]

            found = SO3.rotate_points(Rab, Xb).eval()
            assert np.allclose(expected, found)

    def test_log_theta(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            quat = tf.placeholder(dtype, [ 5, 3, None, 4 ])

            found_tangent, found_theta = SO3.log_theta(quat)
            assert found_tangent.shape.as_list() == [ 5, 3, None, 3 ]
            assert found_theta.shape.as_list() == [ 5, 3, None, 1 ]

            found_tangent = SO3.log(quat)
            assert found_tangent.shape.as_list() == [ 5, 3, None, 3 ]

        with tf.Session() as session:

            # Test operation on single omega.

            quat = [ -0.11800829, 0.0068667947, -0.26173874, 0.9578725 ]
            expected_tangent = [ -0.23938771, 0.013929753, -0.53095456 ]
            expected_theta = [ 0.58259185 ]

            found_tangent, found_theta = SO3.log_theta(quat)
            assert np.allclose(expected_tangent, found_tangent.eval())
            assert np.allclose(expected_theta, found_theta.eval())

            found_tangent = SO3.log(quat)
            assert np.allclose(expected_tangent, found_tangent.eval())

            # Test operation on multiple omegas.

            quat = [[ -0.11800829,  0.00686680, -0.26173874, 0.9578725 ],
                    [  0.02892915, -0.10760084,  0.19253964, 0.9749429 ]]

            expected_tangent = [[ -0.23938771,  0.01392975, -0.53095456 ],
                                [  0.05834645, -0.21701733,  0.38832818 ]]

            expected_theta = [[0.58259185], [0.44866425]]

            found_tangent, found_theta = SO3.log_theta(quat)
            assert np.allclose(expected_tangent, found_tangent.eval())
            assert np.allclose(expected_theta, found_theta.eval())

            found_tangent = SO3.log(quat)
            assert np.allclose(expected_tangent, found_tangent.eval())

    def test_exp_theta(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            omega = tf.placeholder(dtype, [ 5, 3, None, 3 ])

            found_quat, found_theta = SO3.exp_theta(omega)
            assert found_quat.shape.as_list() == [ 5, 3, None, 4 ]
            assert found_theta.shape.as_list() == [ 5, 3, None, 1 ]

            found_quat = SO3.exp(omega)
            assert found_quat.shape.as_list() == [ 5, 3, None, 4 ]

        with tf.Session() as session:

            # Test operation on single omega.

            omega = [ -0.53021266, 0.36258249, -0.67966044 ]
            expected_quat = [ -0.25555124, 0.17475706, -0.3275819, 0.89266099 ]
            expected_theta = 0.935163

            found_quat, found_theta = SO3.exp_theta(omega)
            assert np.allclose(expected_quat, found_quat.eval())
            assert np.allclose(expected_theta, found_theta.eval())

            found_quat = SO3.exp(omega)
            assert np.allclose(expected_quat, found_quat.eval())

            # Test operation on multiple omegas.

            omega = [[ -0.53021266,  0.36258249, -0.67966044 ],
                     [  0.06892461, -0.21322593,  0.28874894 ]]

            expected_quat = [[ -0.2555512,  0.1747571, -0.3275819, 0.8926610 ],
                             [  0.0342708, -0.1060205,  0.1435722, 0.9833474 ]]

            expected_theta = [[ 0.935163 ], [0.36550219]]

            found_quat, found_theta = SO3.exp_theta(omega)
            assert np.allclose(expected_quat, found_quat.eval())
            assert np.allclose(expected_theta, found_theta.eval())

            found_quat = SO3.exp(omega)
            assert np.allclose(expected_quat, found_quat.eval())

    def test_hat(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            omega = tf.placeholder(dtype, [ 5, 3, None, 3 ])
            expected = tf.placeholder(dtype, [ 5, 3, None, 3, 3 ])
            found = SO3.hat(omega)
            assert expected.shape.as_list() == found.shape.as_list()

        with tf.Session() as session:

            # Test operation on single omega.

            omega = [ 0.0689246, -0.213226, 0.288749 ]

            expected = [[ 0.000000, -0.288749, -0.213226 ],
                        [ 0.288749,  0.000000, -0.068925 ],
                        [ 0.213226,  0.068925,  0.000000 ]]

            found = SO3.hat(omega).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple omegas.

            omega = [[  0.06892461, -0.21322593,  0.28874894 ],
                     [ -0.53021266,  0.36258249, -0.67966044 ]]

            expected = [[[  0.00000000, -0.28874894, -0.21322593 ],
                         [  0.28874894,  0.00000000, -0.06892461 ],
                         [  0.21322593,  0.06892461,  0.00000000 ]],
                        [[  0.00000000,  0.67966044,  0.36258249 ],
                         [ -0.67966044,  0.00000000,  0.53021266 ],
                         [ -0.36258249, -0.53021266,  0.00000000 ]]]

            found = SO3.hat(omega).eval()
            assert np.allclose(expected, found)

class TestSE3:
    
    def test_inverse(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            se3 = tf.placeholder(dtype, [ 5, 3, None, 7 ])
            inverse = SE3.inverse(se3)
            assert se3.shape.as_list() == inverse.shape.as_list()

        with tf.Session() as session:

            # Test operation on single transform.

            translation = [ -0.1, +0.2, -0.3 ]
            quat = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            se3 = tf.concat((translation, quat), -1)

            expected = [ 0.10411537, -0.20916086, 0.29225284, -0.034270799,
                    0.10602051, -0.14357218, 0.98334744 ]

            found = SE3.inverse(se3).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple transforms.

            translation = [[ -0.1, +0.2, -0.3 ],
                           [ +0.4, -0.5, +0.6 ]]

            quat = [[  0.03427079, -0.10602051,  0.14357218, 0.98334744 ],
                    [ -0.25555124,  0.17475706, -0.32758190, 0.89266099 ]]

            se3 = tf.concat((translation, quat), -1)

            expected = [
                    [ 0.10411537, -0.20916086, 0.29225284, -0.034270799,
                        0.10602051, -0.14357218, 0.98334744 ],
                    [-0.5400572, 0.47161696, -0.50588108, 0.25555124,
                        -0.17475706, 0.3275819, 0.89266099 ]]

            found = SE3.inverse(se3).eval()
            assert np.allclose(expected, found)

    def test_transform_transform(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            Tab = tf.placeholder(dtype, [ 5, 3, None, 7 ])
            Tbc = tf.placeholder(dtype, [ 5, 3, None, 7 ])
            found = SE3.transform_transform(Tab, Tbc)
            assert Tab.shape.as_list() == found.shape.as_list()

        with tf.Session() as session:

            # Test operation on single transform pair.

            translation = [ -0.1, +0.2, -0.3 ]
            quat = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            Tab = tf.concat((translation, quat), -1)

            translation = [ +0.4, -0.5, +0.6 ]
            quat = [ -0.25555124,  0.17475706, -0.32758190, 0.89266099 ]
            Tbc = tf.concat((translation, quat), -1)

            expected = [ 0.30013049, -0.22688024, 0.35396395, -0.21106331,
                    0.05174298, -0.21507015, 0.95211332 ]

            found = SE3.transform_transform(Tab, Tbc).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple transform pairs.

            translation = [ -0.1, +0.2, -0.3 ]
            quat = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            T1 = tf.concat((translation, quat), -1)

            translation = [ +0.4, -0.5, +0.6 ]
            quat = [ -0.25555124,  0.17475706, -0.32758190, 0.89266099 ]
            T2 = tf.concat((translation, quat), -1)

            expected = [
                    [ 0.30013049, -0.22688024, 0.35396395, -0.21106331,
                        0.05174298, -0.21507015, 0.95211332 ],
                    [ 0.2828464, -0.40415477, 0.2578177, -0.2303436,
                        0.10267009, -0.17286093, 0.95211332 ]]

            Tab = tf.stack((T1, T2), -2)
            Tbc = tf.stack((T2, T1), -2)
            found = SE3.transform_transform(Tab, Tbc).eval()
            assert np.allclose(expected, found)

    def test_transform_transforms(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            Tab = tf.placeholder(dtype, [ 5, 3, None, 7 ])
            Tbc = tf.placeholder(dtype, [ 5, 3, None, 8, 7 ])
            found = SE3.transform_transforms(Tab, Tbc)
            assert Tbc.shape.as_list() == found.shape.as_list()

        with tf.Session() as session:

            # Test operation on single transform pair.

            translation = [ -0.1, +0.2, -0.3 ]
            quat = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            T1 = tf.concat((translation, quat), -1)

            translation = [ +0.4, -0.5, +0.6 ]
            quat = [ -0.25555124,  0.17475706, -0.32758190, 0.89266099 ]
            T2 = tf.concat((translation, quat), -1)

            expected = [
                    [ 0.30013049, -0.22688024, 0.35396395, -0.21106331,
                        0.05174298, -0.21507015, 0.95211332 ],
                    [ -0.19195443, 0.39312845, -0.60699476, 0.067400204,
                        -0.20851, 0.28236266, 0.93394439 ]]

            Tab = T1
            Tbc = tf.stack((T2, T1), -2)
            found = SE3.transform_transforms(Tab, Tbc).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple transform pairs.

            translation = [ -0.1, +0.2, -0.3 ]
            quat = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            T1 = tf.concat((translation, quat), -1)

            translation = [ +0.4, -0.5, +0.6 ]
            quat = [ -0.25555124,  0.17475706, -0.32758190, 0.89266099 ]
            T2 = tf.concat((translation, quat), -1)

            expected = [
                    [[ 0.30013049, -0.22688024, 0.35396395, -0.21106331,
                         0.05174298, -0.21507015, 0.95211332 ],
                     [ -0.19195443, 0.39312845, -0.60699476, 0.067400204,
                         -0.20851, 0.28236266, 0.93394439 ]],
                    [[ 0.2828464, -0.40415477, 0.2578177, -0.2303436,
                         0.10267009, -0.17286093, 0.95211332 ],
                     [ 0.72961519, -0.89199879, 1.3125242, -0.45624124,
                         0.31199762, -0.58483916, 0.59368727 ]]]

            T3 = tf.stack((T2, T1), -2)
            T4 = tf.stack((T1, T2), -2)

            Tab = tf.stack((T1, T2), -2)
            Tbc = tf.stack((T3, T4), -2)

            found = SE3.transform_transforms(Tab, Tbc).eval()
            assert np.allclose(expected, found)

    def test_transform_point(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            Tab = tf.placeholder(dtype, [ 5, 3, None, 7 ])
            Xb = tf.placeholder(dtype, [ 5, 3, None, 3 ])
            found = SE3.transform_point(Tab, Xb)
            assert Xb.shape.as_list() == found.shape.as_list()

        with tf.Session() as session:

            # Test operation on single transform-point pair.

            translation = [ -0.1, +0.2, -0.3 ]
            quat = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            Tab = tf.concat((translation, quat), -1)

            Xb = [ +0.4, -0.5, +0.6 ]

            expected = [ 0.30013049, -0.22688024, 0.35396395 ]

            found = SE3.transform_point(Tab, Xb).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple transform-point pairs.

            translation = [ -0.1, +0.2, -0.3 ]
            quat = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            T1 = tf.concat((translation, quat), -1)

            translation = [ +0.4, -0.5, +0.6 ]
            quat = [ -0.25555124,  0.17475706, -0.32758190, 0.89266099 ]
            T2 = tf.concat((translation, quat), -1)

            Xb = [[ +0.4, -0.5, +0.6 ],
                  [ -0.1, +0.2, -0.3 ]]

            expected = [[ 0.30013049, -0.22688024, 0.35396395 ],
                        [ 0.28284640, -0.40415477, 0.25781770 ]]

            Tab = tf.stack((T1, T2), -2)
            found = SE3.transform_point(Tab, Xb).eval()
            assert np.allclose(expected, found)

    def test_transform_points(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            Tab = tf.placeholder(dtype, [ 5, 3, None, 7 ])
            Xb = tf.placeholder(dtype, [ 5, 3, None, 8, 3 ])
            found = SE3.transform_points(Tab, Xb)
            assert Xb.shape.as_list() == found.shape.as_list()

        with tf.Session() as session:

            # Test operation on single transform-point pair.

            translation = [ -0.1, +0.2, -0.3 ]
            quat = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            Tab = tf.concat((translation, quat), -1)

            Xb = [[ +0.4, -0.5, +0.6 ],
                  [ -0.1, +0.2, -0.3 ]]

            expected = [
                    [  0.30013049, -0.22688024,  0.35396395 ],
                    [ -0.19195443,  0.39312845, -0.60699476 ]]

            found = SE3.transform_points(Tab, Xb).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple transform-point pairs.

            translation = [ -0.1, +0.2, -0.3 ]
            quat = [ 0.034270799, -0.10602051, 0.14357218, 0.98334744 ]
            T1 = tf.concat((translation, quat), -1)

            translation = [ +0.4, -0.5, +0.6 ]
            quat = [ -0.25555124,  0.17475706, -0.32758190, 0.89266099 ]
            T2 = tf.concat((translation, quat), -1)

            Xb = [[[ +0.4, -0.5, +0.6 ],
                   [ -0.1, +0.2, -0.3 ]],
                  [[ -0.1, +0.2, -0.3 ],
                   [ +0.4, -0.5, +0.6 ]]]

            expected = [[[  0.30013049, -0.22688024,  0.35396395 ],
                         [ -0.19195443,  0.39312845, -0.60699476 ]],
                        [[  0.28284640, -0.40415477,  0.25781770 ],
                         [  0.72961519, -0.89199879,  1.31252420 ]]]

            Tab = tf.stack((T1, T2), -2)
            found = SE3.transform_points(Tab, Xb).eval()
            assert np.allclose(expected, found)

    def test_log_theta(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            se3 = tf.placeholder(dtype, [ 5, 3, None, 7 ])
            found_tangent = SE3.log(se3)
            assert found_tangent.shape.as_list() == [ 5, 3, None, 6 ]

        with tf.Session() as session:

            # Test operation on single omega.

            se3 = [ -0.5745919, 0.11275912, -0.30823658, 0.028929152,
                    -0.10760084, 0.19253964, 0.9749429 ]

            expected_tangent = [ -0.57734011, 0.21665844, -0.24975949,
                    0.058346454, -0.21701733, 0.38832818 ]

            found_tangent = SE3.log(se3)
            assert np.allclose(expected_tangent, found_tangent.eval())

            # Test operation on multiple omegas.

            se3 = [[ -0.5745919, 0.11275912, -0.30823658, 0.028929152,
                     -0.10760084, 0.19253964, 0.9749429 ],
                   [ -0.5745919, 0.11275912, -0.30823658, -0.11800829,
                      0.0068667947, -0.26173874, 0.9578725 ]]

            expected_tangent = [
                    [ -0.57734011, 0.21665844, -0.24975949, 0.058346454,
                        -0.21701733, 0.38832818 ],
                    [ -0.59210997, -0.0057419919, -0.30344724, -0.23938771,
                        0.013929753, -0.53095456 ]]

            found_tangent = SE3.log(se3)
            assert np.allclose(expected_tangent, found_tangent.eval())

    def test_exp(self):

        # Test unknown shapes with various data-types.

        dtypes = [ tf.float16, tf.float32, tf.float64 ]

        for dtype in dtypes:

            tangent = tf.placeholder(dtype, [ 5, 3, None, 6 ])
            found = SE3.exp(tangent)
            assert found.shape.as_list() == [ 5, 3, None, 7 ]

        with tf.Session() as session:

            # Test operation on single transform-point pair.

            tangent = [ 0.9349634, -1.021437, 1.083277, -1.0604253, 0.72516498,
                    -1.3593209 ]

            expected = [ 0.72961515, -0.89199883, 1.3125242, -0.45624124,
                    0.31199763, -0.58483917, 0.59368726 ]

            found = SE3.exp(tangent).eval()
            assert np.allclose(expected, found)

            # Test operation on multiple transform-point pairs.

            tangent = [[  0.9349634, -1.021437, 1.083277, -1.0604253,
                            0.72516498, -1.3593209 ],
                       [ -0.59764871, -0.047360125, -0.29555033, -0.32238547,
                           -0.0097570063, -0.70907191 ]]

            expected = [[  0.72961515, -0.89199883, 1.3125242, -0.45624124,
                             0.31199763, -0.58483917, 0.59368726 ],
                        [ -0.5745919, 0.11275912, -0.30823658, -0.15714798,
                            -0.0047560884, -0.34563969, 0.92510275 ]]

            found = SE3.exp(tangent).eval()
            assert np.allclose(expected, found)

