#coding: utf-8
'''
author: SoonyenJu
Data: 2016-10-19
Version: 2nd Version
'''
from pyhdf.HDF import *
from pyhdf.V   import *
from pyhdf.VS  import *
from pyhdf.SD  import *
import numpy as np
import pandas as pd
import scipy as sp
import os, csv

def main():
    from scipy import linalg, optimize, interpolate
    work_dir = r"G:\Pys_HCHO_Workshop\Algorithms\OMI\DATA\DATA_out"
    os.chdir(work_dir)
    amfdir = r"G:\Pys_HCHO_Workshop\Algorithms\OMI\DATA\DATA_in\amf"
    refdir = work_dir + "\\refit.csv"
    if os.path.isfile(refdir) == False: datprep()
    refit = pd.read_csv(work_dir + "\\refit.csv").values[:, 1:]
    xscs = pd.read_csv(work_dir + "\\xscs.csv")
    speran = xscs["spc"].values
    # hcho scd :{
    x = np.vstack([xscs["hcho"], xscs["bro"], xscs["oclo"], xscs["o3"]]).T
    #test: {
    print "for now, it's ok'"
    # }
    # test: {
    '''
    scds = np.abs(np.array([linalg.lstsq(x, refit[i, :])[0] for i in range(refit.shape[0])]))
    '''
    scds = np.empty([refit.shape[0], 4])
    for i in range(refit.shape[0]):
        try:
            r = refit[i, np.where(np.isfinite(refit[i, :]))][0, :]
            x_ = x[np.where(np.isfinite(refit[i, :])), :][0, :, :]
            scds[i, :] = linalg.lstsq(x_, r)[0]
            # print scds[i, :]
            print 111111111111111111111
        except:
            print 2222222222222222222
            print np.where(np.isnan(refit[i, :]) == False)
            scds[i, :] = np.ones(4) * np.float("nan")
    # }
    hcho = scds[:, 0]
    del(scds); del(xscs); del(refit)
    # }
    # hcho vcd: {
    sza_model = np.array([87, 87.1, 87.2, 87.3, 87.4, 87.5, 87.6, 87.7, 87.8, 87.9, 88, 88.1, 88.2, 88.3,
					88.4, 88.5, 88.6, 88.7, 88.8, 88.9, 89, 89.1, 89.2, 89.3, 89.4, 89.5, 89.6])
    vza_model = np.array([0, 5, 10, 15, 20, 25, 35, 45, 55, 60, 65])
    # cal amf and its wavelength: {
    amf = np.loadtxt(amfdir + "\\amf.dat")
    amf_wavlen = np.array([amf[i, :][0] for i in range(amf.shape[0])])
    amf = np.array([amf[i, :][1:] for i in range(amf.shape[0])])
    l = np.abs(amf_wavlen - speran[0]).argmin()
    r = np.abs(amf_wavlen - speran[-1]).argmin()
    amf = np.mean(amf[l: r, :], axis = 0).reshape(table_size)
    # amf = pd.DataFrame(amf, index = sza_model, columns = vza_model)

    info = pd.read_csv(work_dir + "\\info.csv")
    sza = info["sza"].values; vza = info["vza"].values
    vcd = np.array([hcho[i]/amf[np.abs(sza[i] - sza_model).argmin(), \
                np.abs(vza[i] - vza_model).argmin()] for i in range(hcho.shape[0])])
    del(sza); del(vza)
    result = pd.DataFrame(np.vstack([hcho, vcd]).T, columns = (["scd", "vcd"]))
    result["lat"] = info["lat"]; result["lon"] = info["lon"]; result.to_csv("retrieval.csv")
    del(info); del(result); del(hcho); del(vcd)
    os.remove("info.csv"); os.remove("refit.csv"); os.remove("xscs.csv");    table_size = [27, 11]

    # }
    # }

def datprep():
    work_dir = r"G:\Pys_HCHO_Workshop\Algorithms\OMI\DATA\DATA_out"
    xsc_dir = r"G:\Pys_HCHO_Workshop\Algorithms\OMI\DATA\DATA_in\xsc"
    hdir = xsc_dir + "\\H2CO.txt"; bdir = xsc_dir + "\\BrO.txt";
    cdir = xsc_dir + "\\ClO2.txt"; odir = xsc_dir + "\\O3.txt";
    lon_ran = [111.3, 115.5]; lat_ran = [21.5, 24.5];
    #test: {
    lon_ran = [90, 180]; lat_ran = [10, 90];
    #}
    speran = np.linspace(328.5, 342.5, np.abs(342.5 - 328.5)/0.05)
    #Technically, speran should be 328.5nm to 356.5nm
    os.chdir(work_dir)
    in_dir = r"G:\Pys_HCHO_Workshop\DATA\DATA_OMI_DOAS\DATA_in\050921\\"
    rdir = in_dir + "OMI-Aura_L1-OML1BRUG_2005m0921t0529-o06305_v003-2011m0120t195711-p1.he4"
    idir = in_dir + "OMI-Aura_L1-OML1BIRR_2005m0921t2337-o06316_v003-2007m0417t023751.he4"
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
    # update: 1018{
    flag = np.ones(pos.shape[0])
    np.savetxt("flag.txt", flag)
    # }
    lat = lat[pos]; lon = lon[pos]
    # calculate refraction: {
    radman = h4read(rdir, refs_dict["RadianceMantissa"]).reshape([c1*c2, c3])
    radman = radman[pos, :]
    # update: 1018{
    coor = np.vstack([lon, lat]).T
    pointer = np.copy(coor)
    flag[np.where(radman < 0)[0]] = 0
    # }
    radexp = h4read(rdir, refs_dict["RadianceExponent"]).reshape([c1*c2, c3])
    radexp = radexp[pos, :]
    # update: 1019{
    '''
    rad = radman * (10 ** radexp); del(radman); del(radexp)
    '''
    #}


    irrman = h4read(idir, irefs_dict["IrradianceMantissa"]).reshape([c2, c3])
    irrman = irrman[pos_irr, :]
     # update: 1018{
    flag[np.where(irrman<0)[0]] = 0
    # }
    irrexp = h4read(idir, irefs_dict["IrradianceExponent"]).reshape([c2, c3])
    irrexp = irrexp[pos_irr, :]
   # update: 1019 {
    '''
    irr = irrman * (10 ** irrexp); del(irrman); del(irrexp)
    '''
    '''
    p = np.unique(np.where(radman > 0)[0])
    print p.shape
    print np.unique(np.where(irrman[p] > 0))
    # p = p[np.where(irrman[p] > 0)]
    print p.shape
    raw_input('')
    rad = radman[p] * (10 ** radexp[p]); del(radman); del(radexp)
    irr = irrman[p] * (10 ** irrexp[p]); del(irrman); del(irrexp)
    pointer = pointer[p]
    raw_input('')
    rad = flagdata(radman, flag) * (10 ** flagdata(radexp, flag)); del(radman); del(radexp)
    irr = flagdata(irrman, flag) * (10 ** flagdata(irrexp, flag)); del(irrman); del(irrexp)  
    '''
    radman[np.where(radman < 0)] = np.float("nan")
    irrman[np.where(irrman < 0)] = np.float("nan")
    rad = radman * (10 ** radexp); del(radman); del(radexp)
    irr = irrman * (10 ** irrexp); del(irrman); del(irrexp)
    #}

    sza = h4read(rdir, refs_dict["SolarZenithAngle"]).ravel(); sza = sza[pos]
    cosza = np.abs(np.cos(sza)) # I don't know if it's right, is there any possibility that sza could be greater than 90 degrees?
    # update: 1019{
    '''
    refra = np.array([rad[i, :] / (irr[i, :] * cosza[i]) \
                    for i in range(len(pos))]) * np.pi
    '''
    '''
    cosza = flagdata(cosza, flag)
    lat_ = flagdata(lat, flag); lon_ = flagdata(lon, flag)
    refra = np.array([rad[i, :] / (irr[i, :] * cosza[i]) \
                    for i in range(len(np.where(flag == 1)[0]))]) * np.pi
    cond_pos = np.where((refra < 0) | (refra > 1))
    lat_[cond_pos]; lon_[cond_pos]
    raw_input('')
    '''
    refra = np.array([rad[i, :] / (irr[i, :] * cosza[i]) \
                for i in range(len(pos))]) * np.pi
    #}
    del(cosza)
    # } cal ends
    # cal wavelength: {
    wc = h4read(rdir, refs_dict["WavelengthCoefficient"]).reshape([c1*c2, -1])
    wc = wc[pos, :]
    wavlen = cal_wavlen(wc); del(wc)
    # } cal ends
    # test: {
    '''
    interp refra into given spectra: {
    n_pos = np.where(refra < 0)
    refra[n_pos] = refra.max() #guess something wrong!!!!!!!!!!!!!!!!!!!!!!!
    '''
    refra[np.where(refra <0)] = np.float("nan")
    #}
    #test: {
    refit = np.array([spefit(wavlen[i, :], refra[i, :], speran) \
                    for i in range(len(pos))]); del(refra)
    '''
    refit = np.empty([refra.shape[0], 280])
    for i in range(len(pos)):
        try:
            refit[i, :] = spefit(wavlen[i, :], refra[i, :], speran)
        except ValueError as e:
            refit[i, :] = 1
    '''
    #}
    #refit = np.abs(refit) # preventing negtives test！！
    # }
    # differential value: {
    refit = - np.log(refit)
    refit = np.array([polfitdif(speran, refit[i, :]) for i in range(len(pos))])
    # }
    # save refra and info: {
    vza = h4read(rdir, refs_dict["ViewingZenithAngle"]).ravel(); vza = vza[pos]
    info = np.vstack([lat, lon, sza, vza]).T
    info = pd.DataFrame(info, columns = ["lat", "lon", "sza", "vza"]); info.to_csv("info.csv")
    refit = pd.DataFrame(refit); refit.to_csv("refit.csv"); refit = refit.as_matrix()
    del(refit); del(info); del(lat); del(lon); del(sza); del(vza)
    # }
    # cal differential xscs: {
    h = xscprep(hdir, speran); b = xscprep(bdir, speran);
    c = xscprep(cdir, speran); o = xscprep(odir, speran);
    xscs = np.vstack([speran, h, b, c, o]).T
    xscs = pd.DataFrame(xscs, columns = ["spc", "hcho", "bro", "oclo", "o3"]);
    xscs.to_csv("xscs.csv");
    del(xscs)
    # }

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

def cal_wavlen(wavcoef, filename = "wavlen.csv", wavrefcol = 281, wavran = 557):
    wc = wavcoef; del(wavcoef); shape = wc.shape[0]
    wavlenref = np.linspace(1, wavran, wavran) - wavrefcol
    wavlen = np.empty([shape, wavran])

    for i in range(wc.shape[0]):
        wavlen[i, :] = wc[i, 0] + \
                wc[i, 1] * wavlenref + \
                wc[i, 2] * wavlenref**2 + \
                wc[i, 3] * wavlenref**3 + \
                wc[i, 4] * wavlenref**4
    #updata: 1019 {
        if wavlen[i, 0] < 300 or wavlen[i, -1] > 385: wavlen[i, :] = np.float("nan")
    # }
    del(wc); del(wavlenref)
    # writer = csv.writer(open(filename, 'wb'))
    # writer.writerows(wavlen)
    return wavlen

def spefit(x, y, new_x):
    from scipy.interpolate import splev, splrep
    # updata: 1019 {
    '''
    tck = splrep(x, y)
    return splev(new_x, tck)
    '''
    try:
        tck = splrep(x, y)
        return splev(new_x, tck)
    except:
        return np.ones(new_x.shape) * np.float("nan")
    # }

def polfitdif(speran, val):
    # update: 10.19{
    try:
        speran_ = speran[np.where(np.isfinite(val))]
        val_ = val[np.where(np.isfinite(val))]
        fCurve3p = sp.polyfit(speran_, val_, 3)
        # }
        fCurve3 = sp.polyval(fCurve3p, speran)
        dif = val - fCurve3
        return dif
    except:
        return np.ones(val.shape) * np.float("nan")

def xscprep(dir, speran):
    f = np.loadtxt(dir)
    nu, coef = 10000000/f[::-1, 0], f[::-1, 1]
    fit = spefit(nu, coef, speran)
    dif = polfitdif(speran, fit)
    return dif

def flagdata(data, flag):
    '''
     if len(data.shape) == 2:
            data[np.where(flag == 0), :] = float('nan')
    elif len(data.shape) == 1:
        data[np.where(flag == 0)] = float('nan')
    '''
    if len(data.shape) == 2:
        return data[np.where(flag == 1)[0], :]
    elif len(data.shape) == 1:
        return data[np.where(flag == 1)[0]]


if __name__ == '__main__':
    main()
    print "ok"
