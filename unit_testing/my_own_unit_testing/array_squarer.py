import numpy as np

#Define exceptions
class MyError(Exception): pass
class OutOfRangeError(MyError): pass
class NotIntegerError(MyError): pass

def square_array(a):
    '''
    Given a numpy array, this function returns a numpy array with all elements squared
    :param a: a numpy array
    :return: returns the same numpy array but with all elements squared
    '''
    for i in range(0, len(a)):
        for j in range(0, len(a[0])):
            a[i,j] = a[i,j]**2
    return a

def add_one(num):
    return num + 1


def

