#coding: utf-8
'''
author: SoonyenJu
Data: 2016-12-21
Version: 4th Version
'''
from pyhdf.HDF import *
from pyhdf.V   import *
from pyhdf.VS  import *
from pyhdf.SD  import *
from numba import jit
import numpy as np
import pandas as pd
import scipy as sp
import os, csv
import gdal, osr, ogr

def main():
    rdir = "OMI-Aura_L1-OML1BRUG_2005m0921t0529-o06305_v003-2011m0120t195711-p1.he4"
    wrk_dir = r"D:\OMI161221\in"
    os.chdir(wrk_dir)
    # prep(rdir)
    # scd(rdir)
    # create_shp(rdir)
    tiff = gridproj(rdir, mode = "i")
    draw_tif(tiff, name = "hcho.tif")
    # prod_proc()

def prep(rdir):
    refs_dict = h4lookup(rdir)
    c1, c2, c3 = query(rdir, refs_dict["RadianceMantissa"])[2]
    radman = h4read(rdir, refs_dict["RadianceMantissa"])
    radexp = h4read(rdir, refs_dict["RadianceExponent"])
    rad = radman * (10 ** radexp)
    pacRad = np.load("pacificRad.npy")
    sza = h4read(rdir, refs_dict["SolarZenithAngle"])
    refra = np.array([rad[i, :, :] / pacRad for i in range(c1)])
    for i in range(c3): refra[:, :, i] = np.pi * refra[:, :, i] / np.abs(np.cos(sza))
    os.chdir("../out")
    np.save("refra.npy", refra)
    os.chdir("../in")

def scd(rdir):
    speran = [328.5, 356.5]
    os.chdir("../out")
    ref = np.load("refra.npy")
    os.chdir("../in")
    refs_dict = h4lookup(rdir)
    ref[np.where(ref < 0)] = np.float("nan")
    c1, c2, c3 = query(rdir, refs_dict["RadianceMantissa"])[2]
    scd = np.empty((c1, c2, 6))
    rad_wc = h4read(rdir, refs_dict["WavelengthCoefficient"])
    h_coef = np.loadtxt("clip_h2co_300k.txt")
    n_coef = np.loadtxt("clip_no2_220k.txt")
    b_coef = np.loadtxt("clip_bro_228k.txt")
    o34_coef = np.loadtxt("clip_o3_243k.dat")
    o39_coef = np.loadtxt("clip_o3_293k.dat")
    o4_coef = np.loadtxt("clip_o2o2_296k.txt")
    for i in range(c1):
        for j in range(c2):
            rc = rad_wc[i, j, :]
            wl = cal_wavlen(rc)
            l = np.abs(wl - speran[0]).argmin()
            r = np.abs(wl - speran[1]).argmin()
            dif_ref = polfitdif(wl[l: r], ref[i, j, l: r])
            temp = interp_(h_coef[:, 0], h_coef[:, 1], wl[l: r])
            dif_h = polfitdif(wl[l: r], temp)
            temp = interp_(n_coef[:, 0], n_coef[:, 1], wl[l: r])
            dif_n = polfitdif(wl[l: r], temp)
            temp = interp_(b_coef[:, 0], b_coef[:, 1], wl[l: r])
            dif_b = polfitdif(wl[l: r], temp)
            temp = interp_(o34_coef[:, 0], o34_coef[:, 1], wl[l: r])
            dif_o34 = polfitdif(wl[l: r], temp)
            temp = interp_(o39_coef[:, 0], o39_coef[:, 1], wl[l: r])
            dif_o39 = polfitdif(wl[l: r], temp)
            temp = interp_(o4_coef[:, 0], o4_coef[:, 1], wl[l: r])
            dif_o4 = polfitdif(wl[l: r], temp)
            x = np.vstack([dif_h, dif_n, dif_b, dif_o34, dif_o39, dif_o4]).T
            if dif_ref.shape[0] == 0: scds = [0, 0, 0, 0, 0, 0]
            else: scds = lstsquare(x, dif_ref)
            scd[i,j, :] = scds
        print i
    os.chdir("../out")
    np.save("scd.npy", scd)
    os.chdir("../in")

def create_shp(rdir):
    os.chdir("../out")
    scds = np.load("scd.npy")
    data = scds[:, :, 0].ravel() / (10 ** 15)
    os.chdir("../in")
    refs_dict = h4lookup(rdir)
    c1, c2, c3 = query(rdir, refs_dict["RadianceMantissa"])[2]
    lons = h4read(rdir, refs_dict["Longitude"]).ravel()
    lats = h4read(rdir, refs_dict["Latitude"]).ravel()
    os.chdir("../out")

    # data = np.vstack((lons, lats, data)).T
    data = np.vstack((lons, lats, data)).T

    filename = "vcd.shp"
    dr = ogr.GetDriverByName("ESRI Shapefile")
    ds = dr.CreateDataSource(filename)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    geomtype = ogr.wkbPoint

    lyr = ds.CreateLayer("workshop", srs = sr, geom_type = geomtype)
    field = ogr.FieldDefn("data", ogr.OFTReal)
    field.SetWidth(100)
    lyr.CreateField(field)

    for i in range(data.shape[0]):
        feat = ogr.Feature(lyr.GetLayerDefn())
        feat.SetField("data", data[i, 2])

        wkt = "POINT(%f %f)" % (data[i, 0], data[i, 1])

        point = ogr.CreateGeometryFromWkt(wkt)
        feat.SetGeometry(point)
        lyr.CreateFeature(feat)
        feat.Destroy()

def gridproj(rdir, mode = "g"):
    from scipy import interpolate
    os.chdir("../out")
    scds = np.load("scd.npy")
    array = scds[:, :, 0] / (10 ** 15)
    os.chdir("../in")
    refs_dict = h4lookup(rdir)
    c1, c2, c3 = query(rdir, refs_dict["RadianceMantissa"])[2]
    lons = h4read(rdir, refs_dict["Longitude"])
    lats = h4read(rdir, refs_dict["Latitude"])
    os.chdir("../out")
    # array[np.where(array < 0)] = np.float("nan")
    # array[np.where(array > 30)] = np.float("nan")

    lat_tif = np.array([90 - 0.25 * i for i in range(720)])
    lon_tif = np.array([-180 + 0.25 * i for i in range(1440)])
    xi = []
    tiff = np.zeros([720, 1440])
    lon_left = lons[:, 0]; lon_right = lons[:, -1]
    lat_left = lats[:, 0]; lat_right = lats[:, -1]

    idx = lat_left.argsort()
    lat_left = np.sort(lat_left)
    lon_left = np.array([lon_left[i] for i in idx])

    idx = lat_right.argsort()
    lat_right = np.sort(lat_right)
    lon_right = np.array([lon_right[i] for i in idx])

    lon_left_min = lon_left.min(); lon_left_max = lon_left.max();
    lon_right_min = lon_right.min(); lon_right_max = lon_right.max();
    lon_left = interp_(lat_left, lon_left, lat_tif)
    lon_right = interp_(lat_right, lon_right, lat_tif)
    idx_left = np.where(lon_left > lon_left_min) and np.where(lon_left < lon_left_max)
    idx_right = np.where(lon_right > lon_right_min) and np.where(lon_right < lon_right_max)

    idx = np.array(list(set(idx_left[0]).intersection(set(idx_right[0]))))
    pnts_idx = []
    for i in idx:
        a = np.abs(lon_tif - lon_left[i]).argmin()
        b = np.abs(lon_tif - lon_right[i]).argmin()
        a, b = min(a, b), max(a, b)
        pnts_idx.append([i, a, b])
    del(idx)
    pnts = []
    for i in range(len(pnts_idx)):
        for j in range(pnts_idx[i][1], pnts_idx[i][2] + 1):
            pnts.append([pnts_idx[i][0], j])
    pnts = np.array(pnts)

    lats = lats.ravel(); lons = lons.ravel(); array = array.ravel()
    # idx = np.array(list(set(idx_left[0]).intersection(set(idx_right[0]))))
    idx1 = np.where(array > 0)[0]
    idx2 = np.where(array < 100)[0]
    idx = np.array(list(set(idx1).intersection(set(idx2))))
    array = array[idx]
    lats = lats[idx]
    lons = lons[idx]

    if mode == "g":
        lat_ = lat_tif[pnts[:, 0]]
        lon_ = lon_tif[pnts[:, 1]]

        idx = np.where(np.isnan(array) == False)
        array = array[idx]
        lats = lats[idx]
        lons = lons[idx]
        data = interpolate.griddata((lats, lons), array, (lat_, lon_), method='cubic')
        for i in range(len(data)): tiff[pnts[i, 0], pnts[i, 1]] = data[i]
    elif mode == "i":
        count = 0
        for [r, c] in pnts:
            print len(pnts) - count
            idx1 = np.where(np.abs(lats - lat_tif[r]) < 0.125)[0]
            idx2 = np.where(np.abs(lons - lon_tif[c]) < 0.125)[0]
            idx = np.array(list(set(idx1).intersection(set(idx2))))
            # print idx2
            if len(idx) == 0:
                tiff[r, c] = np.float("nan")
            else:
                idx = idx[np.where(np.isfinite(array[idx]))]
                dis = [np.sqrt((lats[i] - lat_tif[r])**2 + (lons[i] - lon_tif[c])**2) for i in idx]
                p_dis = [i**-2 for i in dis]
                sp_dis = np.sum(p_dis)
                w = [i / sp_dis for i in p_dis]
                tiff[r, c] = np.sum(array[idx] * w)
            count += 1
    return tiff

def draw_tif(tiff, name = "scd0d25.tif"):
    cols = 1440
    rows = 720
    originX = -180
    originY = 90
    pixelWidth = 0.25
    pixelHeight = 0.25

    driver = gdal.GetDriverByName('GTiff')
    newRasterfn = name
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, -pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(tiff)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def prod_proc():
    import h5py
    pdir = "OMI-Aura_L2-OMHCHO_2005m0921t0529-o06305_v003-2014m0620t065847.he5"
    f = h5py.File(pdir)
    # print f.keys()
    df = "/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields"
    gf = "/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields"
    prod = f[df + "/ColumnAmount"].value
    amf = f[df + "/AirMassFactor"].value
    lons = f[gf + "/Longitude"].value
    lats = f[gf + "/Latitude"].value
    prod = prod / 10**15
    np.save("prod.npy", prod)

# hdf4 functions: {
def h4lookup(path, swath = "Earth UV-2 Swath"):
    '''
    only look-up datasets, ignore vdata and
    "WavelengthReferenceColumn" is that.
    '''
    hdf = HDF(path)
    v = hdf.vgstart()
    s2_vg = v.attach(swath)
    geo_tag, geo_ref = s2_vg.tagrefs()[0]
    dat_tag, dat_ref = s2_vg.tagrefs()[1]
    s2_vg.detach()
    #--------------------------------------------
    # found geoloaction & data fields
    #--------------------------------------------
    geo_vgs = v.attach(geo_ref); dat_vgs = v.attach(dat_ref)
    gvg_tagrefs = geo_vgs.tagrefs(); dvg_tagrefs = dat_vgs.tagrefs()
    geo_vgs.detach(); dat_vgs.detach()
    tagrefs_list = gvg_tagrefs + dvg_tagrefs
    refs_dict = {}
    #--------------------------------------------
    # create dict in which keys are names in hdf and values are refs
    #--------------------------------------------
    sd = SD(path)
    for tr in tagrefs_list:
        tag, ref = tr
        if tag == HC.DFTAG_NDG:
            sds = sd.select(sd.reftoindex(ref))
            refs_dict[sds.info()[0]] = ref
    sds.endaccess(); sd.end(); v.end(); hdf.close()
    return refs_dict

def h4read(path, ref):
    '''
    only capable of reading datasets, vdata is not.
    '''
    sd = SD(path)
    sds = sd.select(sd.reftoindex(ref))
    data = np.float64(sds.get())
    sds.endaccess(); sd.end()
    return data

def query(path, ref):
    sd = SD(path)
    sds = sd.select(sd.reftoindex(ref))
    info = sds.info()
    sds.endaccess(); sd.end()
    return info
# }

def cal_wavlen(wavcoef, wavrefcol = 281, wavran = 557):
    wc = wavcoef; del(wavcoef);
    wavlenref = np.linspace(1, wavran, wavran) - wavrefcol

    wavlen = wc[0] + \
            wc[1] * wavlenref + \
            wc[2] * wavlenref**2 + \
            wc[3] * wavlenref**3 + \
            wc[4] * wavlenref**4
    if wavlen[0] < 300 or wavlen[-1] > 390: wavlen[:] = np.float("nan")
    del(wc); del(wavlenref)
    return wavlen

def interp_(x, y, new_x, mode = "s"):
    from scipy.interpolate import splev, splrep, pchip, UnivariateSpline
    try:
        y = y[np.where(np.isfinite(y))]
        x = x[np.where(np.isfinite(y))]
        if mode == "s":
            tck = splrep(x, y)
            return splev(new_x, tck)
        elif mode == "p":
            curve = pchip(x, y)
            return curve(new_x)
        elif mode == "u":
            spline = UnivariateSpline(x, y, k = 3, s = 8)
            return spline(new_x)
    except:
        return np.ones(new_x.shape) * np.float("nan")

def polfitdif(speran, val):
    try:
        speran_ = speran[np.where(np.isfinite(val))]
        val_ = val[np.where(np.isfinite(val))]
        fCurve3p = sp.polyfit(speran_, val_, 3)
        fCurve3 = sp.polyval(fCurve3p, speran)
        dif = val - fCurve3
        return dif
    except:
        return np.ones(val.shape) * np.float("nan")

def lstsquare(x, y):
    from scipy import linalg
    a = np.vstack([x.T, y]).T
    p = np.unique(np.where(~np.isfinite(a))[0])
    x_ = np.delete(x, p, 0)
    y_ = np.delete(y, p, 0)
    try:
        return linalg.lstsq(x_, y_)[0]
    except:
        return np.empty(x.shape[1]) * np.float("nan")

'''
def draw_tif_(rdir):
    from scipy import interpolate
    os.chdir("../out")
    scds = np.load("scd.npy")
    array = scds[:, :, 0] / (10 ** 15)
    os.chdir("../in")
    refs_dict = h4lookup(rdir)
    c1, c2, c3 = query(rdir, refs_dict["RadianceMantissa"])[2]
    lons = h4read(rdir, refs_dict["Longitude"])
    lats = h4read(rdir, refs_dict["Latitude"])
    os.chdir("../out")
    array[np.where(array < 0)] = np.float("nan")
    array[np.where(array > 30)] = np.float("nan")

    lat_tif = np.array([90 - 0.25 * i for i in range(720)])
    lon_tif = np.array([-180 + 0.25 * i for i in range(1440)])
    xi = []
    tiff = np.zeros([720, 1440])
    lon_left = lons[:, 0]; lon_right = lons[:, -1]
    lat_left = lats[:, 0]; lat_right = lats[:, -1]

    idx = lat_left.argsort()
    lat_left = np.sort(lat_left)
    lon_left = np.array([lon_left[i] for i in idx])

    idx = lat_right.argsort()
    lat_right = np.sort(lat_right)
    lon_right = np.array([lon_right[i] for i in idx])

    lon_left_min = lon_left.min(); lon_left_max = lon_left.max();
    lon_right_min = lon_right.min(); lon_right_max = lon_right.max();
    lon_left = interp_(lat_left, lon_left, lat_tif)
    lon_right = interp_(lat_right, lon_right, lat_tif)
    idx_left = np.where(lon_left > lon_left_min) and np.where(lon_left < lon_left_max)
    idx_right = np.where(lon_right > lon_right_min) and np.where(lon_right < lon_right_max)

    idx = np.array(list(set(idx_left[0]).intersection(set(idx_right[0]))))
    pnts_idx = []
    for i in idx:
        a = np.abs(lon_tif - lon_left[i]).argmin()
        b = np.abs(lon_tif - lon_right[i]).argmin()
        a, b = min(a, b), max(a, b)
        pnts_idx.append([i, a, b])
    pnts = []
    for i in range(len(pnts_idx)):
        for j in range(pnts_idx[i][1], pnts_idx[i][2] + 1):
            pnts.append([pnts_idx[i][0], j])
    pnts = np.array(pnts)

    lats = lats.ravel()
    lons = lons.ravel()
    array = array.ravel()
    lat_ = lat_tif[pnts[:, 0]]
    lon_ = lon_tif[pnts[:, 1]]
    # f = np.delete(array, np.where(np.isnan(array) == True))
    # index = np.where(np.isnan(array) == False)
    # array[np.where(np.isnan(array) == True)] = 10
    idx = np.where(np.isnan(array) == False)
    array = array[idx]
    lats = lats[idx]
    lons = lons[idx]
    data = interpolate.griddata((lats, lons), array, (lat_, lon_), method='cubic')
    for i in range(len(data)):
        tiff[pnts[i, 0], pnts[i, 1]] = data[i]

    np.savetxt("data.txt", data)

    cols = 1440
    rows = 720
    originX = -180
    originY = 90
    pixelWidth = 0.25
    pixelHeight = 0.25

    driver = gdal.GetDriverByName('GTiff')
    newRasterfn = "vcd.tif"
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, -pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(tiff)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
'''

if __name__ == '__main__':
    main()
    print "ok"