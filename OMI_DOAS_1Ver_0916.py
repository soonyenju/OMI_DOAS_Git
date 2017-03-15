# coding: utf-8
"""
author: SoonyenJu
Data: 2016-09-19
Version: the 3rd revision
"""
from pyhdf.HDF import *
from pyhdf.V   import *
from pyhdf.VS  import *
from pyhdf.SD  import *

import numpy as np
import pylab as pl
import scipy as sp
from scipy import linalg, optimize, interpolate
import sys, re, os
from math import *
from hapi import *
from scipy.interpolate import splev, splrep
from scipy.ndimage import map_coordinates
from scipy.signal import *
from numpy.polynomial import Chebyshev
from scipy.optimize import *

import gdal, ogr, os, osr 
from numba import jit
import csv

def main():
	out_dir = r"G:\Pys_HCHO_Workshop\Algorithms\OMI\DATA\DATA_out\\"
	in_dir = r"G:\Pys_HCHO_Workshop\DATA\DATA_OMI_DOAS\DATA_in\050921\\"
	rdir = in_dir + "OMI-Aura_L1-OML1BRUG_2005m0921t0529-o06305_v003-2011m0120t195711-p1.he4"
	idir = in_dir + "OMI-Aura_L1-OML1BIRR_2005m0921t2337-o06316_v003-2007m0417t023751.he4"
	xsc_dir = r"G:\Pys_HCHO_Workshop\Algorithms\OMI\DATA\DATA_in\xsc\\"
	amfdir = r"G:\Pys_HCHO_Workshop\Algorithms\OMI\DATA\DATA_in\amf\\"
	hdf4 = HDF4(rdir, idir)
	rad_data, geo_data, irr_data = hdf4.read()
	omi_doas = OMI_DOAS(out_dir, rad_data, geo_data, irr_data, xsc_dir, amfdir)
	

	# cut_refra = np.load(out_dir + "cut_refra.npy")
	# omi_doas.scd_cal(cut_refra)
	# scd = np.load(out_dir + "scd.npy")
	# omi_doas.vcd_cal(scd, amfdir)
	# grid = np.load(out_dir + "vcd.npy")
	# lat = geo_data["Latitude"]; lon = geo_data["Longitude"];
	# omi_doas.save_csv(grid, lat, lon)


class OMI_DOAS(object):
	"""docstring for OMI"""
	def __init__(self, out_dir, rad_data, geo_data, irr_data, xsc_dir, amfdir):
		self.out_dir, self.rad_data, self.geo_data, self.irr_data = out_dir, rad_data, geo_data, irr_data
		self.xsc_dir = xsc_dir; self.amfdir = amfdir		
		self.spec_range = np.linspace(328.5, 342.5, abs(342.5 - 328.5)/0.05)#理论上是328.5到356.5
		self.auto_cal()


	def __del__(self):
		pass
	#---------------------------------------------------------------------------	
	def auto_cal(self):
		self.refra_cal()
		self.refra = np.load(self.out_dir + "refra.npy")	
		self.wavelength_cal()
		# self.cut_refra = np.load(self.out_dir + "cut_refra.npy")

		# self.scd_cal(self.cut_refra)
		# self.scd = np.load(self.out_dir + "scd.npy")
		
		# vcd = self.vcd_cal(self.scd, self.amfdir)	
		# self.lat = self.geo_data["Latitude"]; self.lon = self.geo_data["Longitude"];
		# self.global_gridded(vcd, self.lat, self.lon)

	def refra_cal(self):
		mantissa = self.rad_data["RadianceMantissa"];
		exponent = self.rad_data["RadianceExponent"];
		sza = self.geo_data["SolarZenithAngle"];
		# irradiance = self.irr_data["Irradiance"]; #新数据才有Irradiance
		irr_man = self.irr_data["IrradianceMantissa"]; 
		irr_exp = self.irr_data["IrradianceExponent"];
		irradiance = irr_man*(10**irr_exp)
		
		sza = sza*np.pi/180
		radiance = mantissa*(10**exponent)
		
		temp = pi*radiance
		for index in range(radiance.shape[0]):
			temp[index, :, :] = temp[index, :, :]/irradiance[0, :]

		for index in range(radiance.shape[2]):
			temp[:,:, index] = temp[:,:, index]/sza
		self.refra = temp
		np.save(self.out_dir + "refra.npy", self.refra)

	@jit
	def wavelength_cal(self):
		wrc = self.rad_data["WavelengthReferenceColumn"][0];
		wc = self.rad_data["WavelengthCoefficient"]
		wl = np.empty(self.refra.shape)
		self.cut_refra = np.empty((wc.shape[0], wc.shape[1], self.spec_range.shape[0]))
		wl_ref = np.linspace(1, self.refra.shape[2], self.refra.shape[2]) - wrc
		for i in range(wc.shape[0]):
			for j in range(wc.shape[1]):
				wl[i, j, :] = wc[i, j, 0] + \
					wc[i, j, 1] * wl_ref + \
					wc[i, j, 2] * wl_ref ** 2 + \
					wc[i, j, 3] * wl_ref ** 3 + \
					wc[i, j, 4] * wl_ref ** 4
				self.cut_refra[i, j, :] = self.interp(wl[i, j, :], self.refra[i, j, :], self.spec_range)
			print "the " + str(i) + "'s loop"
		np.save(self.out_dir + "cut_refra.npy", self.cut_refra)

	def interp(self, x, y, new_x):
		tck = splrep(x, y)
		return splev(new_x, tck)

	def xsc_proc(self, file, spec_range):
		nu, coef = read_xsect(file)
		nu = 10000000/nu		
		nu = nu.tolist(); coef = coef.tolist()
		nu.reverse(); coef.reverse()
		nu = np.array(nu); coef = np.array(coef)
		left = np.abs(nu - spec_range[0]).argmin()
		right = np.abs(nu - spec_range[-1]).argmin()
		nu_ = nu[left: right + 1]; coef_ = coef[left: right + 1]
		# new_coef = self.interp(nu, coef, spec_range) #没用了，以后还要改
		# dif_coef = self.spec_fit(spec_range, new_coef)
		fCurve3p = sp.polyfit(nu_, coef_, 3)
		fCurve3 = sp.polyval(fCurve3p, nu_)
		dif_coef = coef_ - fCurve3
		dif_coef = self.interp(nu_, dif_coef, spec_range)
		return dif_coef

	def spec_fit(self, spec_range, d1v):
		fCurve3p = sp.polyfit(spec_range, d1v, 3)
		fCurve3 = sp.polyval(fCurve3p, spec_range)
		return d1v - fCurve3


	def draw_xsc(self):
		pl.plot(self.spec_range, self.coef_h, label='h2co')
		pl.plot(self.spec_range, self.coef_b, label='bro')
		pl.plot(self.spec_range, self.coef_c, label='oclo')
		pl.plot(self.spec_range, self.coef_3, label='o3')
		pl.plot(self.spec_range, self.coef_4, label='o4')

		pl.legend()
		pl.show()


	def scd_cal(self, cut_refra):
		self.coef_h = self.xsc_proc(self.xsc_dir + 'H2CO.txt', self.spec_range); 
		self.coef_b = self.xsc_proc(self.xsc_dir + 'BrO.txt', self.spec_range); 
		self.coef_c = self.xsc_proc(self.xsc_dir + 'ClO2.txt', self.spec_range);
		self.coef_3 = self.xsc_proc(self.xsc_dir + 'O3.txt', self.spec_range);
		self.coef_4 = self.xsc_proc(self.xsc_dir + 'O4.txt', self.spec_range);
		self.draw_xsc()

		spec_range = self.spec_range
		cut_refra[np.where(cut_refra<0)] = 1
		refra = -np.log(cut_refra)
		refra[np.where(refra<=0)] = -9999
		#end of pre-processing
		c1 = refra.shape[0]; c2 = refra.shape[1]; c3 = refra.shape[2]
		scd_poly = np.zeros((c1, c2, c3), dtype = np.float64)
		scd = np.zeros((c1, c2), dtype = np.float64)
	
		print "start loop"

		
		r = refra[0, 0, :]
		# r = medfilt(r, 3)
		fCurve3p = sp.polyfit(spec_range, r, 3)
		fCurve3 = sp.polyval(fCurve3p, spec_range)
		diff = r - fCurve3
		# cheby = Chebyshev.fit(spec_range, r, 20)
		# c = cheby(spec_range)
		# diff = r - c

		pl.plot(spec_range, r, label = 'original')
		pl.plot(spec_range, fCurve3, label = 'fit')
		pl.plot(spec_range, diff, label = 'diff')
		# pl.plot(spec_range, c, label = 'cheby')

		pl.legend()
		pl.show()
		
		y = diff#/1000
		self.y = y
		print diff
		x = np.vstack([self.coef_h, self.coef_b, self.coef_c, self.coef_3, self.coef_4]).T
		result = linalg.lstsq(x, y)[0]
		np.save(self.out_dir + "x.npy", x)
		np.save(self.out_dir + "y.npy", y)
		np.save(self.out_dir + "sr.npy", spec_range)
		print result
		# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.basinhopping.html
	# 	minimizer_kwargs = {"method": "BFGS"}
	# 	x0 = [10**15, 10**18]
	# 	ret = basinhopping(self.res, x0, minimizer_kwargs=minimizer_kwargs, niter=200)
	# def res(self, x):
	# 	h,o3 = x
	# 	df = np.zeros(2)
	# 	return self.y - (h * self.coef_h + o3 * self.coef_3), df

		
		
		for i in range(c1):
			for j in range(c2):
				spec_range = self.spec_range
				coef_h = self.coef_h
				coef_b = self.coef_b
				coef_c = self.coef_c
				coef_3 = self.coef_3
				coef_4 = self.coef_4
				temp = refra[i, j, :]
				pos = np.where(temp>=0)
				temp = np.take(temp, pos[0])			
				spec_range = np.take(spec_range, pos[0])
				coef_h = np.take(coef_h, pos[0])
				coef_b = np.take(coef_b, pos[0])
				coef_c = np.take(coef_c, pos[0])
				coef_3 = np.take(coef_3, pos[0])
				coef_4 = np.take(coef_4, pos[0])
				if temp.shape[0] == 0:
					scd[i, j] = -9999
					continue
				fCurve3p = sp.polyfit(spec_range, temp, 3)
				fCurve3 = sp.polyval(fCurve3p, spec_range)
				fCurve3 = np.array(map(lambda i: np.array(str(i)).astype(float), fCurve3))
				temp = np.array(map(lambda i: np.array(str(i)).astype(float), temp))
				y = temp - fCurve3		
				# cheby = Chebyshev.fit(spec_range, temp, 55)
				# c = cheby(spec_range)
				# y = temp - c
				x = np.vstack([coef_h, coef_b, coef_c, coef_3, coef_4]).T
				x = np.vstack([coef_h, coef_b, coef_c, coef_3]).T
				# method 1-----------------------------------------------------------------
				result = linalg.lstsq(x, y)[0]
				print result
				scd[i, j] = result[0]
				print scd[i, j]
				print (i, j)
				# method 2-----------------------------------------------------------------
				# p0 = np.array([10**15, 10**13, 10**13, 10**18, 10**-2], dtype = np.float64)
				# result = optimize.leastsq(self.residuals, p0, args = (y, x.ravel()))
				# print result
				# scd[i, j] = result[0][0]
				# print (i, j)
		self.scd = self.neg_eli(scd, mode = "IDW")
		np.save(self.out_dir + "scd.npy", self.scd)	
		print scd
		


	def func(self, x, p):
		ocho, bro, oclo, o3, o4 = p
		x = x.reshape(-1, 5)
		return ocho * x[:, 0] + bro * x[:, 1] + oclo * x[:, 2] + o3 * x[:, 3] + o4 * x[:, 4]

	def residuals(self, p, y, x):
		return y - self.func(x, p)



	def vcd_cal(self, scd, amfdir):
		self.vza = self.geo_data["ViewingZenithAngle"];
		self.sza = self.geo_data["SolarZenithAngle"];
		sza_model = np.array([87, 87.1, 87.2, 87.3, 87.4, 87.5, 87.6, 87.7, 87.8, 87.9, 88, 88.1, 88.2, 88.3,
						88.4, 88.5, 88.6, 88.7, 88.8, 88.9, 89, 89.1, 89.2, 89.3, 89.4, 89.5, 89.6])
		vza_model = np.array([0, 5, 10, 15, 20, 25, 35, 45, 55, 60, 65])
		table_size = [27, 11]

		f = open(amfdir + "amf.dat", 'r')
		data = [line.strip() for line in f.readlines()]; f.close()
		data = data[2:]
		amf_dict = {}

		for i in range(len(data)):
			data[i] = np.array(data[i].split(), dtype = np.float64)
			amf_dict[str(data[i][0])] = data[i][1:].reshape(table_size)
		
		keys = np.array(sorted(amf_dict.keys()), dtype = np.float64)
		# print self.spec_range.shape, keys.shape
		leng = self.spec_range.shape[0]; size = [leng] + table_size
		cut_amf = np.empty(size)
		lengs = [np.abs(self.spec_range[index] - keys).argmin() for index in range(len(self.spec_range))]
		for index in range(len(lengs)):
			cut_amf[index] = amf_dict[str(keys[index])]
		# print cut_amf
		amf_ = np.empty([size[0], (scd.shape)[0]])
		for i in range(size[0]):
			amf0 = cut_amf[i]
			sza0 = self.sza[i,:]; vza0 = self.vza[i,:]
			for j in range(len(sza0)):
				c0 = np.abs(sza_model - sza0[j]).argmin()
				c1 = np.abs(vza_model - vza0[j]).argmin()
				amf_[i, j] = amf0[c0, c1]
		# print amf_.shape
		amf = np.array([amf_[:, i].mean() for i in range(amf_.shape[1])])
		coor = scd.shape
		vcd = np.empty(coor)
		# print coor
		for i in range(coor[1]):
			vcd[:, i] = scd[:, i]/amf[i]
		np.save(self.out_dir + "vcd.npy", vcd)
		return vcd


	def neg_eli(self, d3v, mode = "easy"):
		if len(d3v.shape) == 2: d3v = d3v.reshape([i for i in d3v.shape] + [1])
		if mode == "easy":
			c = d3v.shape
			d3v = d3v.ravel()
			a = np.where(d3v < 0)
			for i in a:
				indices = [i-1, i+1]
				d3v[i] = np.take(d3v, indices, mode = 'clip').mean()
			d3v = d3v.reshape(c)
		elif mode == "IDW":
			x, y, z = np.where(d3v < 0)
			for k in range(len(x)):
				d2v = d3v[:,:,z[k]]
				indices = self.window_creator(x[k], y[k])
				line = self.transfer_ind(indices[1])
				I = indices[1]
				X = [i[0] for i in I]; Y = [i[1] for i in I]
				dis_x = [(x[k] - i)**2 for i in X]
				dis_y = [(y[k] - j)**2 for j in Y]
				dis = [np.sqrt(np.sqrt(dis_x[i] + dis_y[i])) for i in range(len(dis_x))]
				inx, w = self.IDW_weight(dis)
				clip = np.take(d2v, line, mode = 'clip')
				index = [i for i in inx if X[i] >= 0 and Y[i] >= 0]

				interp_val = clip[index] * np.array(w)[index]
				d3v[x[k], y[k], z[k]] = sum(interp_val)			
		return d3v



	def save_csv(self, array, lat, lon, name = 'csv_vcd'):
		vcd = array/10**15
		lat = lat.ravel(); lon = lon.ravel(); vcd = vcd.ravel()

		csvfile = file(self.out_dir + name + '.csv', 'wb')
		writer = csv.writer(csvfile)
		writer.writerow(['lon', 'lat', 'vcd'])
		data = []
		for i in range(vcd.shape[0]):
			data.append([lon[i], lat[i], vcd[i]])
		writer.writerows(data)
		csvfile.close()	


	def global_gridded(self, vcd, lat, lon):
		lats = [lat[0, 0], lat[-1, 0], lat[0, -1], lat[-1, -1]]
		lons = [lon[0, 0], lon[-1, 0], lon[0, -1], lon[-1, -1]]
		latMax = max(lats); lonMax = max(lons)
		latMin = min(lats); lonMin = min(lons)
		leftup = [-180, 90]; leftdown = [-180, -90]
		rightup = [180, 90]; rightdown = [180, -90]

		pixelWidth = 0.25; pixelHeight = 0.25
		global_grid = np.zeros([180/0.25, 360/0.25])
		# global_lon = np.empty([180/0.25, 360/0.25])
		# global_lat = np.empty([180/0.25, 360/0.25])
		print global_grid.shape
		global_lon = np.linspace(-180, 180, 1440)
		global_lat = np.linspace(-90, 90, 720)
		
		lon_list = [ i for i in np.where((global_lon - lonMax)<0)[0] \
			if i in np.where((global_lon - lonMin)>0)[0]]
		lat_list = [ i for i in np.where((global_lat - latMax)<0)[0] \
			if i in np.where((global_lat - latMin)>0)[0]]	
		# data = {lon_list[0]: lat_list}
		# print data
		coors_list = [[[i, j] for j in lon_list] for i in lat_list]

		dif = np.abs(lon - 110) + np.abs(lat - 20)
		pos = np.where(dif == dif.min())
		x = pos[0]; y = pos[1]
		print lat[x, y], lon[x, y]
		print x, y	
		
		line_list = []
		for count, l in enumerate(coors_list):
			print count
			line_list = line_list + l
		print "done"

		for count, inx in enumerate(line_list):
			print (len(line_list) - count)/1000
			local_lon = -180 + 0.25 * inx[1]
			local_lat = -90 + 0.25 * inx[0]
			# print inx[0], inx[1]
			# print local_lon, local_lat
			if (latMin < local_lat < latMax) and (lonMin < local_lon < lonMax):
				global_grid[inx[0], inx[1]] = self.IDW_interp(local_lat, local_lon,lat, lon, vcd, dt = 7)
				print global_grid[inx[0], inx[1]]
		np.save(self.out_dir + "global_grid.npy", global_grid)


	@jit	
	def IDW_interp(self, latV, lonV, lat, lon, data, dt = 7):
		dif = np.abs(lon - lonV) + np.abs(lat - latV)
		pos = np.where(dif == dif.min())
		x = pos[0]; y = pos[1]
		# print lat[x, y], lon[x, y]
		indices = self.window_creator(x[0], y[0])
		line = self.transfer_ind(indices[0])
		d = self.geo_distance(latV, lonV, lat, lon, line)
		d = np.array(d)
		index = (np.where(d < dt)[0]).tolist()
		# print index
		# d = d[index]
		inx, w = self.IDW_weight(d)
		vcd_clip = np.take(data, line, mode = 'clip')
		interp_val = vcd_clip[index] * np.array(w)[index]
		interp_val = sum(interp_val)
		return interp_val



	def transfer_ind(self, indices, x_len = 60):
		line = [i[0]*x_len + i[1] for i in indices]
		return line

		
	def window_creator(self, x0, y0, size = 3):
		indices = []
		l = np.floor(size/2)
		rangeI = [i - l for i in range(size)]
		rangeJ = [j - l for j in range(size)]
		for i in rangeI:
			for j in rangeJ:
				indices.append([x0 + i, y0 + j])
		mid = indices.index([x0, y0])
		indices_ = indices[:]
		indices_.pop(mid)
		return indices, indices_

	def geo_distance(self, lat0, lon0, lat, lon, line_idx):
		lons = np.take(lon, line_idx, mode = 'clip')
		lats = np.take(lat, line_idx, mode = 'clip')
		dis_lon = [(lon0 - j)**2 for j in lons]
		dis_lat = [(lat0 - i)**2 for i in lats]
		dis = [np.sqrt(np.sqrt(dis_lon[i] + dis_lat[i])) for i in range(len(dis_lon))]
		return dis


	def IDW_weight(self, distance ,p = 4):
		idw = [i**-p for i in distance]
		index, weight = [], []
		for inx, val in enumerate(idw):
			w = round(idw[inx]/sum(idw), 2)
			index.append(inx); weight.append(w)
		return index, weight		

	def draw_global_tif(self, gridded_vcd):
		array = gridded_vcd
		array = array/(10**15)
		array = array[::-1, :]
		cols = 1440
		rows = 720
		originX = -180
		originY = 90
		pixelWidth = 0.25
		pixelHeight = 0.25

		driver = gdal.GetDriverByName('GTiff')
		newRasterfn = self.out_dir + "vcd_global.tif"
		outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
		outRaster.SetGeoTransform((originX, pixelWidth, 0, -originY, 0, pixelHeight))
		outband = outRaster.GetRasterBand(1)
		outband.WriteArray(array)
		outRasterSRS = osr.SpatialReference()
		outRasterSRS.ImportFromEPSG(4326)
		outRaster.SetProjection(outRasterSRS.ExportToWkt())
		outband.FlushCache()



class HDF4(object):
	"""docstring for Hdf4"""
	def __init__(self, rdir, idir):
		self.rdir = rdir
		self.idir = idir
	def __del__(self):
		pass
		
	def read(self):
		rad_m = ("RadianceMantissa", "RadianceExponent", 
			"WavelengthCoefficient", "WavelengthReferenceColumn")
		geo_m = ("Latitude", "Longitude", "SolarZenithAngle", "ViewingZenithAngle")
		# irr_m = ("Irradiance", "IrradianceMantissa") #老数据无Irradiance，只能自己计算
		irr_m = ("IrradianceMantissa", "IrradianceExponent")
		swath_r = "Earth UV-2 Swath"
		swath_i = "Sun Volume UV-2 Swath"	
		r_lp = self.hdf4lookup(self.rdir, swath_r)
		i_lp = self.hdf4lookup(self.idir, swath_i)
		rad_data = self.look(self.rdir, rad_m, r_lp)
		geo_data = self.look(self.rdir, geo_m, r_lp)
		irr_data = self.look(self.idir, irr_m, i_lp)
		print "HDF4 data read in successful."
		return rad_data, geo_data, irr_data
	
	def hdf4lookup(self, path, swath):
		hdf = HDF(path)
		sd = SD(path)
		vs = hdf.vstart()
		v  = hdf.vgstart()

		vg = v.attach(swath)
		vg_members = vg.tagrefs()
		vg0_members = {}
		for tag, ref in vg_members:
			vg0 = v.attach(ref)
			if tag == HC.DFTAG_VG:	
				vg0_members[vg0._name] = vg0.tagrefs()
			vg0.detach
		vg.detach
		
		lookup_dict = {}
		for key in vg0_members.keys():
			for tag, ref in vg0_members[key]:
				if tag == HC.DFTAG_NDG:
					# f = open(swath + '.txt', 'a'); f.writelines('#' + key + '#' + '\n'); f.close()
					sds = sd.select(sd.reftoindex(ref))
					name = sds.info()[0]
					lookup_dict[name] = [tag, ref]
					sds.endaccess()
				elif tag == HC.DFTAG_VH:
					vd = vs.attach(ref)
					nrecs, intmode, fields, size, name = vd.inquire()
					lookup_dict[name] = [tag, ref]
		v.end()
		vs.end()
		sd.end()
		return lookup_dict

	def look(self, path , mem_list, lp_list):
		data = {}
		for name in mem_list:# subdata sets type data
			tag = lp_list[name][0]
			ref = lp_list[name][1]
			if tag == HC.DFTAG_NDG:
				sd = SD(path)
				sds = sd.select(sd.reftoindex(ref))
				data[name] = np.float64(sds.get())
				sds.endaccess()
				sd.end()
			elif tag == HC.DFTAG_VH: #vd type data
				hdf = HDF(path)
				vs = hdf.vstart()
				vd = vs.attach(ref)
				nrecs, intmode, fields, size, name = vd.inquire()
				data[name] = np.full(nrecs, np.float64(vd.read()[0]))
				vs.end()
				hdf.close()

		return data

if __name__ == '__main__':
	main()
	print "ok"