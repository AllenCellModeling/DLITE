# returns best fit radius of a circular arc to the data set (x,y)

from scipy import optimize
import numpy as np

method_2 = "leastsq"

def fit(x,y):

	def calc_R(xc, yc):
	    """ calculate the distance of each 2D points from the center (xc, yc) """
	    return np.sqrt((x-xc)**2 + (y-yc)**2)

	def f_2(c):
	    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
	    Ri = calc_R(*c)
	    return Ri - Ri.mean()

	x_m = np.mean(x)
	y_m = np.mean(y)

	center_estimate = x_m, y_m
	center_2, ier = optimize.leastsq(f_2, center_estimate)

	xc_2, yc_2 = center_2
	Ri_2       = calc_R(*center_2)
	R_2        = Ri_2.mean()
	residu_2   = np.sum((Ri_2 - R_2)**2)

	theta1 = np.rad2deg(np.arctan2(y[np.argmax(x)]-yc_2, x[np.argmax(x)]-xc_2)) # starting angle
	theta2 = np.rad2deg(np.arctan2(y[np.argmin(x)]-yc_2, x[np.argmin(x)]-xc_2)) 

	return R_2, xc_2, yc_2