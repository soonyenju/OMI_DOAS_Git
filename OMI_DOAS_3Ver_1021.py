#coding: utf-8
'''
author: SoonyenJu
Data: 2016-10-21
Version: 3rd Version
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

def main(work_dir, r_name, i_name):
    os.chdir(work_dir)
    preparation(r_name, i_name)
    scd()
    vcd()
    draw_tif()

def preparation(r_name, i_name):
    lon_ran = [111.3, 115.5]; lat_ran = [21.5, 24.5];
    lon_ran = [-180, 180]; lat_ran = [-90, 90]
    speran = np.linspace(328.5, 356.5, np.abs(356.5 - 356.5)/0.05)
    #Technically, speran should be 328.5nm to 356.5nm
    os.chdir("in")
    rdir = r_name
    idir = i_name
    # get look-up dict with names and refs: {
    refs_dict = h4lookup(rdir); irefs_dict = h4lookup(idir, swath = "Sun Volume UV-2 Swath")
    c1, c2, c3 = query(rdir, refs_dict["RadianceMantissa"])[2]
    lat = h4read(rdir, refs_dict["Latitude"]).ravel()
    lon = h4read(rdir, refs_dict["Longitude"]).ravel()
    # }

    # get positions covering Pearl Delta only: {
    # See: https://www.douban.com/note/341056688/; for advanced np.where usage
    pos = np.where((lat_ran[0] < lat) & (lat < lat_ran[1]) \
                    & (lon_ran[0] < lon) & (lon < lon_ran[1]))[0]
    pos_irr = [divmod(p, c2)[1] for p in pos]
    # } finished
    # neccessary info{
    lon = lon[pos]; lat = lat[pos];
    sza = h4read(rdir, refs_dict["SolarZenithAngle"]).ravel()[pos]
    vza = h4read(rdir, refs_dict["ViewingZenithAngle"]).ravel()[pos]
    info = pd.DataFrame(np.vstack([lat, lon, sza, vza]).T, columns = ["lat", "lon", "sza", "vza"])
    rad_wc = h4read(rdir, refs_dict["WavelengthCoefficient"]).reshape([c1*c2, -1])[pos, :]
    irr_wc = h4read(idir, irefs_dict["WavelengthCoefficient"]).reshape([c2, 5])[pos_irr, :]
    #}
    # read in xsc files: {
    hcho = np.loadtxt("hcho_clip.txt")
    bro = np.loadtxt("bro_clip.txt")
    oclo = np.loadtxt("oclo_clip.txt")
    o3 = np.loadtxt("o3_clip.txt")
    o4 = np.loadtxt("o4_clip.txt")
    # }
    # rad & irr{
    radman = h4read(rdir, refs_dict["RadianceMantissa"]).reshape([c1*c2, c3])[pos, :]
    radexp = h4read(rdir, refs_dict["RadianceExponent"]).reshape([c1*c2, c3])[pos, :]
    irrman = h4read(idir, irefs_dict["IrradianceMantissa"]).reshape([c2, c3])[pos_irr, :]
    irrexp = h4read(idir, irefs_dict["IrradianceExponent"]).reshape([c2, c3])[pos_irr, :]
    rad = radman * (10 ** radexp)
    irr = irrman * (10 ** irrexp)
    # }

    # save data: {
    os.chdir("../out")
    info.to_csv("info.csv")
    np.save("rad_wc.npy", rad_wc)
    np.save("irr_wc.npy", irr_wc)
    np.save("rad.npy", rad)
    np.save("irr.npy", irr)
    np.save("hcho.npy", hcho)
    np.save("bro.npy", bro)
    np.save("oclo.npy", oclo)
    np.save("o3.npy", o3)
    np.save("o4.npy", o4)
    # }

    # preprocess of raw xsc files, prototype code: {
    # hcho = np.loadtxt("H2CO.txt")[::-1]
    # hcho[:, 0] = 10000000/hcho[:, 0]
    # p = np.where((325.0 < hcho[:, 0]) & (hcho[:, 0] < 360.0))[0]
    # hcho = hcho[p, :]
    # np.savetxt("hcho_clip.txt", hcho)
    # }
    os.chdir("..")

def scd():
    os.chdir("out")
    rad = np.load("rad.npy"); irr = np.load("irr.npy")
    rad_wc = np.load("rad_wc.npy"); irr_wc = np.load("irr_wc.npy");
    hcho = np.load("hcho.npy")
    bro = np.load("bro.npy"); oclo = np.load("oclo.npy")
    o3 = np.load("o3.npy"); o4 = np.load("o4.npy")
    sza = pd.read_csv("info.csv")["sza"].values
    rad[np.where(rad < 0)] = np.float("nan")
    irr[np.where(irr < 0)] = np.float("nan")
    cosza = np.abs(np.cos(sza)); del(sza)
    scd = np.empty(rad.shape[0])


    for i in range(rad.shape[0]):
    # for i in range(100):
        r = rad[i, :]; f = irr[i, :]; rc = rad_wc[i, :]; ic = irr_wc[i, :]
        r_wl = cal_wavlen(rc); i_wl = cal_wavlen(ic);
        start = np.abs(r_wl - 328.5).argmin()
        end = np.abs(r_wl - 356.5).argmin()
        spr = r_wl[start: end]; del(r_wl)
        r = r[start: end]
        f = interp_(i_wl, f, spr)
        r = (np.pi * r) / (cosza[i] * f); del(f); del(i_wl)
        # 有很多负数， 插值后的吸收截面
        h = interp_(hcho[:, 0], hcho[:, 1], spr)
        b = interp_(bro[:, 0], bro[:, 1], spr); c = interp_(oclo[:, 0], oclo[:, 1], spr)
        o = interp_(o3[:, 0], o3[:, 1], spr); e = interp_(o4[:, 0], o4[:, 1], spr)
        h[np.where(h < 0)] = np.float("nan")
        b[np.where(b < 0)] = np.float("nan")
        c[np.where(c < 0)] = np.float("nan")
        o[np.where(o < 0)] = np.float("nan")
        e[np.where(e < 0)] = np.float("nan")
        #-----------------------------------------------
        r = polfitdif(spr, r)
        h = polfitdif(spr, h)
        b = polfitdif(spr, b); c = polfitdif(spr, c);
        o = polfitdif(spr, o); e = polfitdif(spr, e);
        x = np.vstack([h, b, c, o]).T

        scds = lstsquare(x, r)
        print scds
        print i
        scd[i] = scds[0]
    np.save("scd.npy", scd)
    os.chdir("..")

def vcd():
    sza_model = np.array([87, 87.1, 87.2, 87.3, 87.4, 87.5, 87.6, 87.7, 87.8, 87.9, 88, 88.1, 88.2, 88.3,
                    88.4, 88.5, 88.6, 88.7, 88.8, 88.9, 89, 89.1, 89.2, 89.3, 89.4, 89.5, 89.6])
    vza_model = np.array([0, 5, 10, 15, 20, 25, 35, 45, 55, 60, 65])
    table_size = [27, 11]

    os.chdir("in")
    # cal amf and its wavelength: {
    amf = np.loadtxt("amf.dat")
    os.chdir("../out")
    scd = np.load("scd.npy")
    amf_wavlen = np.array([amf[i, :][0] for i in range(amf.shape[0])])
    amf = np.array([amf[i, :][1:] for i in range(amf.shape[0])])
    l = np.abs(amf_wavlen - 328.5).argmin()
    r = np.abs(amf_wavlen - 356.5).argmin()
    amf = np.mean(amf[l: r, :], axis = 0).reshape(table_size)
    # amf = pd.DataFrame(amf, index = sza_model, columns = vza_model)

    info = pd.read_csv("info.csv")
    sza = info["sza"].values; vza = info["vza"].values
    vcd = np.array([scd[i]/amf[np.abs(sza[i] - sza_model).argmin(), \
                np.abs(vza[i] - vza_model).argmin()] for i in range(scd.shape[0])])
    del(sza); del(vza)
    result = pd.DataFrame(np.vstack([scd, vcd]).T, columns = (["scd", "vcd"]))
    result["lat"] = info["lat"]; result["lon"] = info["lon"]; result.to_csv("retrieval.csv")
    np.save("vcd.npy", vcd
        )
    del(info); del(result); del(scd); del(vcd)
    os.chdir("..")

def draw_tif():
    import gdal, ogr, osr, os
    from scipy import interpolate

    os.chdir("out")
    csvfile = pd.read_csv("retrieval.csv")
    array = csvfile["vcd"].values/(10 ** 15)
    array = np.abs(array)
    array[np.where(array > 100)] = np.float("nan")
    # array[np.where(array > 100)] = -9999
    lat = csvfile["lat"].values
    lon = csvfile["lon"].values
    coor = np.vstack([lon, lat]).T

    # Parse a delimited text file of volcano data and create a shapefile
    # use a dictionary reader so we can access by field name
    # set up the shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")

    # create the data source
    data_source = driver.CreateDataSource("doas_hcho.shp")

    # create the spatial reference, WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # create the layer
    layer = data_source.CreateLayer("hcho", srs, ogr.wkbPoint)

    # Add the fields we're interested in
    layer.CreateField(ogr.FieldDefn("Latitude", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("Longitude", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("vcd", ogr.OFTReal))

    # Process the text file and add the attributes and features to the shapefile
    for i in range(coor.shape[0]):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        pointCoor = coor[i]
        feature.SetField("Latitude", pointCoor[1])
        feature.SetField("Longitude", pointCoor[0])
        feature.SetField("vcd", array[i])

        # create the WKT for the feature using Python string formatting
        wkt = "POINT(%f %f)" %  (float(pointCoor[0]) , float(pointCoor[1]))

        # Create the point from the Well Known Txt
        point = ogr.CreateGeometryFromWkt(wkt)

        # Set the feature geometry using the point
        feature.SetGeometry(point)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Destroy the feature to free resources
        feature.Destroy()

    x_min, x_max, y_min, y_max = layer.GetExtent()
    x_range = np.ceil((x_max - x_min)/0.25)
    y_range = np.ceil((y_max - y_min)/0.25)
    tx = np.linspace(x_min, x_max, x_range)
    ty = np.linspace(y_min, y_max, y_range)
    XI, YI = np.meshgrid(tx, ty)
    array[np.where(array == -9999)] = np.float("nan")
    ZI = interpolate.griddata((lon, lat), array, (XI, YI), method = "linear")

    cols = np.int(x_range)
    rows = np.int(y_range)
    originX = x_min
    originY = y_min
    pixelWidth = 0.25
    pixelHeight = 0.25

    driver = gdal.GetDriverByName('GTiff')
    newRasterfn = "vcd.tif"
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(ZI)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

    # Destroy the data source to free resources
    data_source.Destroy()


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

if __name__ == '__main__':
    main()
    print "ok"