import pytest
import numpy as np
import array_squarer


class TestKnownValues():
    def test_add_one_known_values(self):
        '''The function should return the same array as np.power(a, 2)'''
        for i in range(0, 100):
            assert i+1 == array_squarer.add_one(i)
    def test_array_squarer_known_values(self):
        '''The function should return the same array as np.power(a, 2)'''
        for i in range(0, 100):
            a = np.random.rand(3, 3)
            assert np.array_equiv(np.power(a, 2), array_squarer.square_array(a))

