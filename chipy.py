import dcpy.util
import hdf5storage as hs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import xarray as xr


class chipod:
    """ Base class for chipod instruments.
        Depends heavily on output from chipod_gust routines.
        Represents chipod + ancillary data associated with that
        particular chipod
    """

    def __init__(self, basedir, unit, chifile='Turb.mat', best='', depth=0):
        self.basedir = basedir
        self.unit = unit
        self.name = 'χ-' + unit + ' | ' + str(depth) + ' m'

        # setup dirs for chipod_gust
        self.inputdir = basedir + unit + '/input/'
        self.procdir = basedir + unit + '/proc/'
        if 'Turb' in chifile:
            self.chifile = basedir + unit + '/proc/' + chifile
        else:
            self.chifile = basedir + unit + '/proc/combined/' + chifile

        self.depth = depth
        self.χestimates = []
        self.time = []
        self.best = best
        self.dt = []
        self.Tchi = []
        self.pitot = None
        self.Tzi = None

        self.season = dict()
        self.events = dict()

        self.mixing_seasons = dict()

        # import os
        # this lets me import sally's processed output
        # if not os.path.isdir(self.chidir):
        #    self.chidir = basedir

        # nearby mooring instruments
        self.ctd1 = dict()
        self.ctd2 = dict()
        self.adcp = dict([])

        # derived quantities
        self.turb = dict()
        self.chi = dict()
        self.KT = dict()
        self.Jq = dict()

        # read in χ and calculate derived quantities
        self.LoadChiEstimates()
        self.LoadTzi()
        self.LoadCTD()

        # self.convert_to_xarray()

    def convert_to_xarray(self, estimate='best'):
        ''' Makes xarray dataset of best estimate. '''

        if estimate == 'best':
            chi = self.turb[self.best]
        else:
            chi = self.turb[estimate]

        try:
            names = chi.dtype.names
        except AttributeError:
            names = chi

        dims = ['time']
        coords = {'time': dcpy.util.mdatenum2dt64(chi['time'])}

        turb = xr.Dataset()
        for name in names:
            if ('time' in name or 'spec' in name
                or name == 'nfft' or name == 'stats'
                    or name == 'wda' or name == 'sign_used'):
                continue

            turb[name] = xr.DataArray(chi[name].squeeze(),
                                      dims=dims, coords=coords)

        return turb

    def LoadCTD(self):
        ''' Loads data from proc/T_m.mat '''

        if self.ctd1 == dict() or self.ctd2 == dict():
            import hdf5storage as hs

            mat = hs.loadmat(self.procdir + '/T_m.mat',
                             struct_as_record=False, squeeze_me=True)
            self.ctd1 = mat['T1']
            self.ctd2 = mat['T2']
            self.ctd1.time = self.ctd1.time - 366
            self.ctd2.time = self.ctd2.time - 366

    def LoadT1T2(self):
        ''' Loads data from internal χ-pod sensors '''
        mat = hs.loadmat(self.procdir + '/temp.mat',

                         struct_as_record=False, squeeze_me=True)

        self.Tchi = mat['T']
        self.Tchi['time'] = self.Tchi['time'] - 366

    def LoadTzi(self):
        ''' Load internal stratification estimate '''
        import os

        path = self.procdir + '../input/dTdz_i.mat'
        if os.path.isfile(path):
            mat = hs.loadmat(path, struct_as_record=False, squeeze_me=True)
            try:
                self.Tzi = mat['Tz_i']
                self.Tzi.time -= 366
            except:
                pass

    def LoadChiEstimates(self):
        ''' Loads all calculated chi estimates using h5py '''

        if not self.turb == dict([]):
            return

        try:
            import h5py
            f = h5py.File(self.chifile, 'r')
        except OSError:
            # not an hdf5 file
            f = sp.io.loadmat(self.chifile)

        def process_field(obj, struct, name):

            self.turb[name] = xr.Dataset()

            try:
                struct = struct[0, 0]
            except ValueError:
                pass

            # convert to dt64
            if 'time' in struct.dtype.names:
                time64 = dcpy.util.mdatenum2dt64(
                    struct['time'].squeeze() - 366)

            for fld in ['eps', 'Jq', 'Kt', 'chi', 'dTdz',
                        'N2', 'T', 'S', 'eps_Kt']:
                if fld not in struct.dtype.names:
                    continue

                self.turb[name][fld] = xr.DataArray(
                    struct[fld].squeeze(), dims=['time'],
                    coords={'time': time64})

            if 'mm' in name:
                Tzmat = sp.io.loadmat(self.basedir + self.unit +
                                      '/input/dTdz_m.mat')
                Sz = xr.DataArray(
                    Tzmat['Tz_m']['Sz'][0, 0][0],
                    dims=['time'],
                    coords={'time': dcpy.util.mdatenum2dt64(
                        Tzmat['Tz_m']['time'][0, 0][0] - 366)})

                self.turb[name]['Sz'] = Sz.interp(time=self.turb[name].time)

        for field in f['Turb'].dtype.names:
            if field in ['mm1', 'mm2', 'pm1', 'pm2',
                         'mi11', 'mi22', 'pi11', 'pi22']:
                name = field
                self.χestimates.append(name)
                process_field(self, f['Turb'][0, 0][field], name)

                if 'wda' in f['Turb'][0, 0][field].dtype.names:
                    process_field(self, f['Turb'][0, 0][field][0, 0]['wda'],
                                  name + 'w')
                    self.χestimates.append(name + 'w')

        self.time = self.turb[self.χestimates[0]]['time']
        self.dt = (self.time[1] - self.time[0]) * 86400  # in seconds

        # average together similar estimates
        for ff in ['mm', 'pm', 'mi', 'pi']:
            self.AverageEstimates(self.turb, ff)
            self.AverageEstimates(self.turb, ff, suffix='w')

        self.chi = self.turb

    def AverageEstimates(self, var, ff, suffix=''):
        ''' Average like estimates in var. '''

        import warnings
        if 'i' in ff:
            e1 = ff + '11' + suffix
            e2 = ff + '22' + suffix
        else:
            e1 = ff + '1' + suffix
            e2 = ff + '2' + suffix

        ff = ff + suffix

        variables = ['chi', 'eps', 'eps_Kt', 'Kt', 'Jq', 'T']
        names = ['$χ$', '$ε$', '$ε$', '$K_T$', '$J_q^t$', '$T$']
        units = ['', 'W/kg', 'W/kg', 'm²/s', 'W/m²', 'C']

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Mean of empty slice')
            if e1 in var and e2 in var:
                var[ff] = xr.Dataset()
                var[ff]['time'] = self.turb[e1]['time']

                self.χestimates.append(ff)
                for vv, nn, uu in zip(variables, names, units):
                    if vv not in var[e1]:
                        continue

                    var[ff][vv] = var[e1][vv].copy(
                        data=np.nanmean([var[e1][vv].values,
                                         var[e2][vv].values], axis=0))
                    var[ff][vv].attrs = {'long_name': nn, 'units': uu}

                var[ff]['dTdz'] = var[e1]['dTdz']
                var[ff]['Sz'] = var[e1]['Sz']
                var[ff]['N2'] = var[e1]['N2']

    def CalcKT(self):
        raise ValueError('CalcKT is deprecated.')

    def CalcJq(self):
        raise ValueError('CalcJq is deprecated.')

    def LoadSallyChiEstimate(self, fname, estname):
        ''' fname - the mat file you want to read from.
            estname - what you want to name the estimate.

            Output saved as self.turb[estname] '''

        import os
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)

        data = sp.io.loadmat(fname)

        chi = data['avgchi']
        self.turb[estname + '1'] = dict()
        self.turb[estname + '2'] = dict()

        for field in chi.dtype.names:
            temp = chi[field][0][0][0]
            self.turb[estname + '1'][field] = temp
            self.turb[estname + '2'][field] = temp

        self.turb[estname + '1']['chi'] = self.turb[estname + '1']['chi1']
        self.turb[estname + '2']['chi'] = self.turb[estname + '2']['chi2']

        self.turb[estname + '1']['time'] -= 366
        self.turb[estname + '2']['time'] -= 366
        self.χestimates.append(estname + '1')
        self.χestimates.append(estname + '2')

    def LoadPitotOld(self):
        ''' Load pitot data from proc/Praw.mat into self.pitot '''

        rawname = self.procdir + '/Praw.mat'
        pitot = hs.loadmat(rawname,
                           squeeze_me=True, struct_as_record=False)
        pitot = pitot['Praw']
        w = pitot['W'][0, 0]
        pitot['W'][0, 0][w > 1] = np.nan
        pitot['W'][0, 0][w < 0.4] = np.nan

        pitot['W'] = pitot['W'][0, 0]
        pitot['time'] = pitot['time'][0, 0]
        self.pitotrange = range(0, len(pitot['time'][0, 0]))
        self.pitot = pitot

    def load_pitot(self):
        rawname = self.inputdir + 'vel_p.mat'

        pitot = hs.loadmat(rawname, squeeze_me=True, struct_as_record=False)
        time = ((-86400 + pitot['vel_p'].time * 86400).astype('timedelta64[s]')
                + np.datetime64('0000-01-01').astype('datetime64[ns]'))
        spd = xr.DataArray(pitot['vel_p'].spd, name='spd',
                           dims=['time'], coords=[time])
        u = xr.DataArray(pitot['vel_p'].u, name='u',
                         dims=['time'], coords=[time])
        v = xr.DataArray(pitot['vel_p'].v, name='v',
                         dims=['time'], coords=[time])

        self.pitot = xr.merge([spd, u, v])

        try:
            shear = xr.DataArray(pitot['vel_p'].shear, name='shear',
                                 dims=['time'], coords=[time])
            self.pitot = xr.merge([shear, self.pitot])
        except AttributeError:
            pass

    def PlotPitotRawVoltage(self, hax=None):
        import dcpy.util

        if hax is None:
            hax = plt.gca()

        pitotrange = self.pitotrange
        hax.hold(True)
        hax.plot_date(dcpy.util.datenum2datetime(
            self.pitot['time'][0, 0][pitotrange]),
            self.pitot['W'][0, 0][pitotrange], '-')
        hax.set_ylabel('Raw Pitot voltage (V)')

    def CompareChipodCTD(self):
        ''' Multipanel plots comparing χ-pod temps with CTD temps '''
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot2grid((4, 2), (0, 0), colspan=2)
        plt.hold(True)
        plt.plot_date(Tctd1.time, Tctd1.T, '-')
        plt.plot_date(Tctd2.time, Tctd2.T, '-')
        plt.plot_date(Tchi.time[chirange], Tchi.T1[chirange], '-')
        plt.plot_date(Tchi.time[chirange], Tchi.T2[chirange], '-')
        plt.legend(["CTD {0:.0f} m".format(Tctd1.z),
                    "CTD {0:.0f} m".format(Tctd2.z),
                    "χ-pod 15 m T₁", "χ-pod 15m T₂"])
        plt.ylabel('Temperature (C)')

        plt.subplot2grid((4, 2), (1, 0), colspan=2)
        plt.hold(True)
        plt.plot_date(Tchi.time[chirange], Tchi.T1[chirange], '-')
        plt.plot_date(Tchi.time[chirange], Tchi.T2[chirange], '-')
        plt.legend(["χ-pod 15 m T₁", "χ-pod 15m T₂"])
        plt.ylabel('Temperature (C)')

        plt.subplot2grid((4, 2), (2, 0))
        plt.plot(Tctd1.T, Tctd2.T, '.')
        plt.xlabel('CTD T at 10m')
        plt.ylabel('CTD T at 20m')
        dcpy.plots.line45()

        plt.subplot2grid((4, 2), (2, 1))
        plt.plot(Tchi.T1[chirange], Tchi.T2[chirange], '.')
        plt.xlabel('χ-pod T₁')
        plt.ylabel('χ-pod T₂')
        dcpy.plots.line45()

        plt.subplot2grid((4, 2), (3, 0))
        T12 = (Tctd1.T + Tctd2.T) / 2
        Tchi12 = np.interp(mpl.dates.date2num(Tctd1.time),
                           mpl.dates.date2num(Tchi.time[chirange]),
                           (Tchi.T1[chirange] + Tchi.T2[chirange]) / 2)
        plt.plot(T12, Tchi12, '.')
        plt.xlabel('CTD (10m + 20m)/2')
        plt.ylabel('χ-pod (T₁ + T₂)/2')
        dcpy.plots.line45()
        plt.grid()
        plt.tight_layout()

    def FilterEstimate(self, kind, time, var, order=1,
                       filter_len=None, decimate=False):

        dt = np.diff(time[0:2]) * 86400
        if filter_len is not None:
            if kind == 'bandpass':
                import dcpy.ts
                from dcpy.util import MovingAverage
                filter_len = np.array(filter_len)
                # runnning average to remove gaps
                var = MovingAverage(
                    var, filter_len.min() / 2 / dt, decimate=False)
                var = dcpy.ts.BandPassButter(var, 1 / filter_len, dt,
                                             order=order)
            elif kind in ['mean', 'median', 'var', 'std', 'Jq']:
                filter_len = np.int(np.floor(filter_len / dt))
                if np.mod(filter_len, 2) == 0:
                    filter_len = filter_len - 1

                import bottleneck as bn
                time = bn.move_mean(time, window=filter_len,
                                    min_count=1)
                if kind == 'mean' or kind == 'Jq':
                    var = bn.move_mean(var, window=filter_len,
                                       min_count=1)

                if kind == 'median':
                    var = bn.move_median(var, window=filter_len,
                                         min_count=1)

                if kind == 'var':
                    var = bn.move_var(var, window=filter_len,
                                      min_count=1)

                if kind == 'std':
                    var = bn.move_std(var, window=filter_len,
                                      min_count=1)

                if decimate is True:
                    # subsample
                    L = filter_len
                    Lb2 = np.int(np.floor(filter_len / 2))
                    time = time[Lb2 + 1::L - 1]
                    var = var[Lb2 + 1::L - 1]

            elif kind == 'hann' or kind is None:
                import dcpy.util
                filter_len = np.int(np.floor(filter_len / dt))
                var = dcpy.util.smooth(var, filter_len, preserve_nan=True)

        return time, var

    def ExtractSeason(self, t, var, name):

        import matplotlib.dates as dt
        from dcpy.util import find_approx

        t = t.copy()
        v = var.copy()

        # NaN out events
        for ss in self.events:
            if 'FW' in ss:
                continue

            ts0, ts1 = self.events[ss]
            is0 = find_approx(t, dt.date2num(ts0))
            is1 = find_approx(t, dt.date2num(ts1))

            t[is0:is1] = np.nan
            v[is0:is1] = np.nan

        # now search for season
        if name in self.season:
            t0, t1 = self.season[name]
            it0 = find_approx(t, dt.date2num(t0))
            it1 = find_approx(t, dt.date2num(t1))

            return t[it0:it1], v[it0:it1]

    def PlotEstimate(self, varname, est, hax=None, filt=None,
                     filter_len=None, tind=None, linewidth=1,
                     decimate=False, **kwargs):

        import dcpy.plots

        self.LoadChiEstimates()

        if hax is None:
            hax = plt.gca()

        try:
            time = self.turb[est]['time'].squeeze()
        except:
            time = self.time

        if tind is None:
            tind = range(len(time))

        if filt is None:
            filt = varname

        var, titlestr, yscale, grdflag = self.ChooseVariable(varname, est)
        if filt == 'bandpass':
            yscale = 'linear'

        time, var = self.FilterEstimate(kind=filt,
                                        time=time[tind], var=var[tind],
                                        filter_len=filter_len,
                                        decimate=decimate)

        hax.plot(time, var, label=est, linewidth=linewidth, **kwargs)
        hax.xaxis_date()
        hax.set_ylabel(titlestr)
        hax.set(yscale=yscale)
        hax.set_ylim(dcpy.plots.robust_lim(var))
        plt.grid(grdflag, axis='y', which='major')

        hax.set_title(titlestr + ' ' + est + self.name)

    def PlotSpectrum(self, varname, est='best', nsmooth=5,
                     filter_len=None, SubsetLength=None,
                     ticks=None, ax=None, norm=False, **kwargs):

        from dcpy.ts import SpectralDensity

        var, titlestr, _, _ = self.ChooseVariable(varname, est)

        t, χ = self.FilterEstimate('mean', self.time, var,
                                   filter_len=filter_len,
                                   decimate=True)
        dt = np.nanmean(np.diff(t) * 86400)

        if SubsetLength is not None:
            SubsetLength /= dt

        S, f, conf = SpectralDensity(χ, dt=dt,
                                     nsmooth=nsmooth,
                                     SubsetLength=SubsetLength,
                                     **kwargs)

        addstr = ''
        if norm:
            normval = np.trapz(S, f)
            S /= normval
            conf /= normval
            addstr = r'/ $\int$ PSD'

        from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
        if ax is None:
            fig = plt.figure()
            ax = SubplotHost(fig, 1, 1, 1)
            fig.add_subplot(ax)

        hdl = ax.loglog(1 / f, S, label=str(self.depth) + ' m')
        if len(conf) > 2:
            ax.fill_between(1 / f, conf[:, 0], conf[:, 1],
                            color=hdl[0].get_color(), alpha=0.3)

        ax.set_ylabel('PSD( ' + titlestr + ' )' + addstr)
        ax.set_xlim([np.min(1 / f), np.max(1 / f)])

        if ticks is not None:
            ax.set_xticks(ticks)
        else:
            ticks = ax.get_xticks()

        if any(ticks > 86400):
            norm = 86400
            tstr = 'days'
        else:
            norm = 3600
            tstr = 'hours'

        tickstr = [str(np.round(xx / norm, 2)) for xx in ticks]
        ax.set_xticklabels(tickstr)
        ax.set_xlabel('Period (' + tstr + ')')
        ax.legend()
        ax.grid(True)

        if not ax.xaxis_inverted():
            ax.invert_xaxis()

        return ax

    def CompareEstimates(self, varname, est1, est2, filter_len=None):

        import dcpy.plots

        var1, titlestr, _, _ = self.ChooseVariable(varname, est1)
        var2, _, _, _ = self.ChooseVariable(varname, est2)

        plt.subplot(2, 2, (1, 2))
        hax = plt.gca()
        self.PlotEstimate(varname, est1, hax, filter_len)
        self.PlotEstimate(varname, est2, hax, filter_len)
        plt.gcf().autofmt_xdate()
        hax.set_title(titlestr + ' | ' + self.name)

        plt.subplot(2, 2, 3)
        hax = plt.gca()
        lv1 = np.log10(var1)
        lv2 = np.log10(var2)
        plt.hist(lv1[np.isfinite(lv1)], bins=40,
                 normed=True, label=est1, histtype='step')
        plt.hist(lv2[np.isfinite(lv2)], bins=40,
                 normed=True, label=est2, histtype='step')
        plt.xlabel(titlestr)
        hax.legend()

        try:
            mask12 = np.isnan(var1) | np.isnan(var2)
        except:
            # should only run when comparing
            # against sally's estimate.
            t1 = self.turb[est1]['time'].squeeze()
            t2 = self.turb[est2]['time'].squeeze()
            var1 = np.interp(t2, t1, var1.squeeze())
            mask12 = np.isnan(var1) | np.isnan(var2)

        var1 = var1[~mask12]
        var2 = var2[~mask12]

        plt.subplot(2, 2, 4)
        hax = plt.gca()
        hax.hexbin(np.log10(var1), np.log10(var2), cmap=plt.cm.YlOrRd)
        hax.set_xlabel(titlestr + '$^{' + est1 + '}$')
        hax.set_ylabel(titlestr + '$^{' + est2 + '}$')
        dcpy.plots.line45()

        plt.tight_layout()
        # lims = [1e-10, 1e-4]
        # plt.xlim(lims); plt.ylim(lims)

    def Summarize(self, est='best', filter_len=None, tind=None,
                  hfig=None, dt=0, debug=False):
        '''
            Summarize χpod deployment.
            Multipanel plots of N², T_z, Tp, T, χ, KT, Jq.
            Can be used to overlay multiple χpods.

            Input:
                 est        : estimate name
                 filter_len : in seconds
                 tind       : subset in time
                 hfig       : plot onto this figure handle
                 dt         : time offset to compare multiple years
                              (first subtracted and then added back
                               so that data is not permanently tampered with)
        '''

        if est == 'best':
            est = self.best

        self.turb[est]['time'] -= dt
        time = self.turb[est]['time']

        if tind is None:
            tind = range(len(time))

        if hfig is None:
            plt.figure(figsize=[8.5, 6])
        else:
            plt.figure(hfig.number)

        if debug is False:
            axN2 = plt.subplot(6, 1, 1)
            ax0 = axN2  # first axis
            axTz = plt.subplot(6, 1, 2, sharex=ax0)
            axT = plt.subplot(6, 1, 3, sharex=ax0)
            axχ = plt.subplot(6, 1, 4, sharex=ax0)
            axKT = plt.subplot(6, 1, 5, sharex=ax0)
            axJq = plt.subplot(6, 1, 6, sharex=ax0)
            ax1 = axJq  # last axis
        else:
            axT = plt.subplot(5, 1, 1)
            ax0 = axT  # first axis
            axacc = plt.subplot(5, 1, 2, sharex=ax0)
            axTP = plt.subplot(5, 1, 3, sharex=ax0)
            axW = plt.subplot(5, 1, 4, sharex=ax0)
            axχ = plt.subplot(5, 1, 5, sharex=ax0)
            ax1 = axχ  # last axis

        if debug is False:
            t, N2 = self.FilterEstimate('mean', time=time[tind],
                                        var=self.turb[est]['N2'][tind],
                                        filter_len=filter_len)
            axN2.plot(t, N2, label=self.name + ' | ' + est, linewidth=0.5)
            axN2.set_ylabel('$N^2$')
            plt.legend()

            t, Tz = self.FilterEstimate('mean', time=time[tind],
                                        var=self.turb[est]['dTdz'][tind],
                                        filter_len=filter_len)
            axTz.plot_date(t, Tz, '-', linewidth=0.5)
            axTz.axhline(0, color='k', zorder=-1)
            axTz.set_ylabel('dT/dz (symlog)')
            axTz.set_yscale('symlog', linthreshy=1e-3, linscaley=0.5)
            axTz.grid(True, axis='y', linewidth=0.5, linestyle='--')
            # if self.Tzi is not None:
            #     ind0 = np.where(self.Tzi.time
            #                     > time[tind][0])[0][0]
            #     ind1 = np.where(self.Tzi.time
            #                     < time[tind][-2])[0][-1]
            #     axTz.plot_date(self.Tzi.time[ind0:ind1],
            #                   self.Tzi.Tz1[ind0:ind1], '-',
            #                   linewidth=0.5)

        if self.Tchi and debug is True:
            if filter_len is None:
                # at minimum, average differentiator
                # to same time resolution as χ estimate
                fl = (time[2] - time[1]) * 86400
            else:
                fl = filter_len

            fl = None
            # T1P
            for tp in ['T1Pt', 'T2Pt']:
                t, v = self.FilterEstimate(None,
                                           time=self.Tchi['time'][0, 0],
                                           var=self.Tchi[tp][0, 0],
                                           filter_len=fl)
                ind0 = np.where(t > time[tind][0])[0][0]
                ind1 = np.where(t < time[tind][-2])[0][-1]
                axTP.plot(t[ind0:ind1], v[ind0:ind1],
                          linewidth=0.25)

            # W
            t, v = self.FilterEstimate(None,
                                       time=self.Tchi['time'][0, 0],
                                       var=self.Tchi['W'][0, 0],
                                       filter_len=fl)
            axW.plot(t[ind0:ind1], v[ind0:ind1],
                     linewidth=0.5)

            # accel
            for acc in ['AX', 'AY', 'AZ']:
                t, v = self.FilterEstimate('mean',
                                           time=self.Tchi['time'][0, 0],
                                           var=self.Tchi[acc][0, 0],
                                           filter_len=filter_len)
                if acc == 'AZ':
                    v = v + 9.81
                axacc.plot(t[ind0:ind1], v[ind0:ind1],
                           linewidth=0.5)

            axTP.set_ylabel('var TP')
            axW.set_ylabel('W')
            axacc.set_ylabel('accel.')
            axacc.set_ylim([-1, 1])

        self.PlotEstimate('T', est=est, decimate=True,
                          linewidth=0.5,
                          filter_len=None, tind=tind,
                          hax=axT)
        axT.set_title('')

        self.PlotEstimate('chi', est=est, decimate=True,
                          filt='median', linewidth=0.5,
                          filter_len=filter_len, tind=tind,
                          hax=axχ)
        axχ.set_title('')

        if debug is False:
            self.PlotEstimate('KT', est=est, decimate=True,
                              filt='median', linewidth=0.5,
                              filter_len=filter_len, tind=tind,
                              hax=axKT)
            axKT.set_title('')

            self.PlotEstimate('Jq', est=est, decimate=True,
                              filt='mean', linewidth=0.5,
                              filter_len=filter_len, tind=tind,
                              hax=axJq)
            axJq.set_title('')
            # axJq.set_yscale('symlog', linthreshy=50, linscaley=2)
            axJq.grid(True, axis='y', linewidth=0.5, linestyle='--')

        ax0.set_xlim([np.nanmin(time[tind]),
                      np.nanmax(time[tind])])
        ax0.xaxis_date()
        plt.gcf().autofmt_xdate()
        if dt != 0:
            import matplotlib.dates as mdates
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        plt.tight_layout(h_pad=-0.1)
        self.turb[est]['time'] += dt

    def SeasonalSummary(self, ax=None, idx: int=0,
                        filter_len: int=86400):

        # seasons = ['NE', 'NE→SW', 'SW', 'SW→NE']
        # cpodcolor = ['indianred', 'slateblue', 'teal', 'darkgreen']
        cpodcolor = mpl.cm.Dark2(np.arange(4))
        handles = []

        # positioning parameters
        n = 1.5
        x0 = 0
        pos = []
        label = []

        if ax is None:
            plt.figure(figsize=[6.5, 4.5])
            ax = plt.gca()
            ax.set_title(self.name)

        var = self.KT[self.best]
        time = self.time
        label.append(self.name)

        time, var = self.FilterEstimate(kind='mean',
                                        time=time, var=var,
                                        filter_len=filter_len,
                                        decimate=True)

        for sidx, ss in enumerate(self.season):
            _, varex = self.ExtractSeason(time, var, ss)
            pos.append(x0 + n * (sidx + 1) + (idx - 0.5) / 3)
            if len(varex) == 0:
                continue
            style = {'color': cpodcolor[idx]}
            meanstyle = {'color': cpodcolor[idx],
                         'marker': '.'}
            hdl = ax.boxplot(np.log10(varex[~np.isnan(varex)]),
                             positions=[pos[-1]],
                             showmeans=True,
                             boxprops=style, medianprops=style,
                             meanprops=meanstyle)
            handles.append(hdl)

        pos = np.sort(np.array(pos))
        ax.set_xticks(pos)

        ax.set_xticklabels(self.season)
        ax.set_xlim([pos[0] - 0.5 - x0, pos[-1] + 0.5])
        ax.set_ylabel('log$_{10} K_T$')
        ax.set_xlabel('season')

        limy = ax.get_yticks()
        limx = ax.get_xticks()
        ax.spines['left'].set_bounds(limy[1], limy[-2])
        ax.spines['bottom'].set_bounds(limx[0], limx[-1])

        ax.legend((handles[0]['medians'][0],
                   handles[-1]['medians'][0]),
                  label)

        return handles[0], label[0], pos

    def ChooseVariable(self, varname, est: str='best'):

        if est == 'best':
            est = self.best

        if varname == 'eps' or varname == 'epsilon' or varname == 'ε':
            var = self.turb[est]['eps'][:].squeeze()
            titlestr = '$ε$'
            yscale = 'log'
            grdflag = True

        if varname == 'chi' or varname == 'χ':
            var = self.turb[est]['chi'][:].squeeze()
            titlestr = '$χ$'
            yscale = 'log'
            grdflag = True

        if varname == 'KT' or varname == 'Kt':
            var = self.KT[est][:].squeeze()
            titlestr = '$K_T$'
            yscale = 'log'
            grdflag = True

        if varname == 'Jq':
            var = self.Jq[est]
            titlestr = '$J_q$'
            yscale = 'linear'
            grdflag = False

        if varname == 'T':
            var = self.turb[est]['T'][:].squeeze()
            titlestr = '$T$'
            yscale = 'linear'
            grdflag = False

        return [var, titlestr, yscale, grdflag]
