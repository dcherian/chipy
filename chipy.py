class chipod:
    """ Base class for chipod instruments.
        Depends heavily on output from chipod_gust routines.
        Represents chipod + ancillary data associated with that
        particular chipod
    """

    def __init__(self, basedir, unit, chifile='Turb.mat', best='', depth=0):
        self.basedir = basedir
        self.unit = unit
        self.name = unit + ' | ' + chifile

        # setup dirs for chipod_gust
        self.inputdir = basedir + unit + '/input/'
        self.procdir = basedir + unit + '/proc/'
        if chifile == 'Turb.mat':
            self.chifile = basedir + unit + '/proc/' + chifile
        else:
            self.chifile = basedir + unit + '/proc/combined/' + chifile

        self.depth = depth
        self.χestimates = []
        self.time = []
        self.best = best
        self.dt = []

        # import os
        # this lets me import sally's processed output
        # if not os.path.isdir(self.chidir):
        #    self.chidir = basedir

        # nearby mooring instruments
        self.ctd1 = dict()
        self.ctd2 = dict()
        self.adcp = dict([])

        # derived quantities
        self.chi = dict()
        self.KT = dict()
        self.Jq = dict()

        # read in χ and calculate derived quantities
        self.LoadChiEstimates()
        self.CalcKT()
        self.CalcJq()

    def LoadCTD(self):
        ''' Loads data from proc/T_m.mat '''

        if self.ctd1 == dict() or self.ctd2 == dict():
            import hdf5storage as hs

            mat = hs.loadmat(self.procdir + '/T_m.mat',
                             struct_as_record=False, squeeze_me=True)
            self.ctd1 = mat['T1']
            self.ctd2 = mat['T2']
            self.ctd1.time = self.ctd1.time - 367
            self.ctd2.time = self.ctd2.time - 367

    def LoadT1T2(self):
        ''' Loads data from internal χ-pod sensors '''
        import scipy.io as spio

        mat = spio.loadmat(self.procdir + '/temp.mat',
                           struct_as_record=False, squeeze_me=True)
        self.Tchi = mat['T']
        self.Tchi.time = self.Tchi.time - 367

    def LoadChiEstimates(self):
        ''' Loads all calculated chi estimates using h5py '''

        # import os
        # import glob

        if not self.chi == dict([]):
            return

        # files = glob.glob(self.chidir + '/*.mat')

        try:
            import h5py
            f = h5py.File(self.chifile, 'r')
        except OSError:
            # not an hdf5 file
            from scipy.io import loadmat
            f = loadmat(self.chifile)

        for field in f['Turb'].dtype.names:
            if field[0:4] == 'chi_':
                name = field[4:]
                self.χestimates.append(name)
            else:
                name = field

            try:
                self.chi[name] = f['Turb'][0, 0][field][0, 0]
            except:
                self.chi[name] = f['Turb'][0, 0][field]

            for fld in ['Jq', 'Kt', 'chi', 'dTdz', 'time', 'N2', 'T', 'S']:
                try:
                    self.chi[name][fld] = self.chi[name][fld][0]
                except:
                    pass

            # convert to matplotline datetime
            try:
                self.chi[name]['time'] = self.chi[name]['time']-367
            except:
                pass

        self.time = self.chi[self.χestimates[0]]['time']
        self.dt = (self.time[1] - self.time[0])*86400  # in seconds

        # average together similar estimates
        for ff in ['mm', 'pm', 'mi', 'pi']:
            self.AverageEstimates(self.chi, ff)

    def AverageEstimates(self, var, ff):
        ''' Average like estimates in var. '''

        import numpy as np

        if 'i' in ff:
            e1 = ff + '11'
            e2 = ff + '22'
        else:
            e1 = ff + '1'
            e2 = ff + '2'

        if e1 in var and e2 in var:
            var[ff] = dict()
            var[ff]['time'] = self.time
            import numpy
            if type(var[e1]) == numpy.void:
                self.χestimates.append(ff)
                var[ff]['chi'] = np.nanmean(
                    [var[e1]['chi'], var[e2]['chi']], axis=0)
                var[ff]['N2'] = var[e1]['N2']
                var[ff]['dTdz'] = var[e1]['dTdz']
            else:  # KT
                var[ff] = np.nanmean(
                    [var[e1], var[e2]], axis=0)

    def CalcKT(self):
        self.LoadChiEstimates()

        for est in self.χestimates:
            if '1' in est or '2' in est:
                # not a combined estimate
                if 'Kt' in self.chi[est]:
                    self.KT[est] = self.chi[est]['Kt'][:]
                else:
                    chi = self.chi[est]['chi'][:]
                    dTdz = self.chi[est]['dTdz'][:]
                    self.KT[est] = (0.5*chi)/dTdz**2

        for ff in ['mm', 'mi', 'pm', 'pi']:
            self.AverageEstimates(self.KT, ff)

    def CalcJq(self):
        import seawater as sw
        import numpy as np

        self.LoadCTD()

        T = (self.ctd1.T + self.ctd2.T)/2
        S = (self.ctd1.S + self.ctd2.S)/2
        P = float(self.depth)

        cp = np.interp(self.time,
                       self.ctd1.time, sw.cp(S, T, P))
        ρ = np.interp(self.time,
                      self.ctd1.time, sw.dens(S, T, P))

        for est in self.χestimates:
            if '1' in est or '2' in est:
                if 'Jq' in self.chi[est]:
                    Jq = self.chi[est]['Jq'][:]
                else:
                    KT = self.KT[est].squeeze()
                    dTdz = self.chi[est]['dTdz'].squeeze()
                    Jq = -ρ * cp * KT * dTdz

                Jq[abs(Jq) > 1000] = np.nan
                self.Jq[est] = Jq

        for ff in ['mm', 'mi', 'pm', 'pi']:
            self.AverageEstimates(self.Jq, ff)

    def LoadSallyChiEstimate(self, fname, estname):
        ''' fname - the mat file you want to read from.
            estname - what you want to name the estimate.

            Output saved as self.chi[estname] '''

        import os
        from scipy.io import loadmat

        if not os.path.exists(fname):
            raise FileNotFoundError(fname)

        data = loadmat(fname)

        chi = data['avgchi']
        self.chi[estname + '1'] = dict()
        self.chi[estname + '2'] = dict()

        for field in chi.dtype.names:
            temp = chi[field][0][0][0]
            self.chi[estname + '1'][field] = temp
            self.chi[estname + '2'][field] = temp

        self.chi[estname + '1']['chi'] = self.chi[estname + '1']['chi1']
        self.chi[estname + '2']['chi'] = self.chi[estname + '2']['chi2']

        self.chi[estname + '1']['time'] = \
                        self.chi[estname + '1']['time'] - 367
        self.chi[estname + '2']['time'] = \
                        self.chi[estname + '2']['time'] - 367
        self.χestimates.append(estname+'1')
        self.χestimates.append(estname+'2')

    def LoadPitot(self):
        ''' Load pitot data from proc/Praw.mat into self.pitot '''
        import numpy as np
        import hdf5storage as hs

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

    def PlotPitotRawVoltage(self, hax=None):
        import matplotlib.pyplot as plt
        import dcpy.util

        if hax is None:
            hax = plt.gca()

        pitotrange = self.pitotrange
        hax.hold(True)
        hax.plot_date(dcpy.util.datenum2datetime(pitot['time'][0,0][pitotrange]),
                      pitot['W'][0,0][pitotrange], '-')
        hax.set_ylabel('Raw Pitot voltage (V)')

    def CompareChipodCTD(self):
        ''' Multipanel plots comparing χ-pod temps with CTD temps '''
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot2grid((4,2), (0,0), colspan=2)
        plt.hold(True)
        plt.plot_date(Tctd1.time, Tctd1.T, '-')
        plt.plot_date(Tctd2.time, Tctd2.T, '-')
        plt.plot_date(Tchi.time[chirange], Tchi.T1[chirange], '-')
        plt.plot_date(Tchi.time[chirange], Tchi.T2[chirange], '-')
        plt.legend(["CTD {0:.0f} m".format(Tctd1.z),
                    "CTD {0:.0f} m".format(Tctd2.z),
                    "χ-pod 15 m T₁", "χ-pod 15m T₂"])
        plt.ylabel('Temperature (C)')

        plt.subplot2grid((4,2), (1,0), colspan=2)
        plt.hold(True)
        plt.plot_date(Tchi.time[chirange], Tchi.T1[chirange], '-')
        plt.plot_date(Tchi.time[chirange], Tchi.T2[chirange], '-')
        plt.legend(["χ-pod 15 m T₁", "χ-pod 15m T₂"])
        plt.ylabel('Temperature (C)')

        plt.subplot2grid((4,2),(2,0))
        plt.plot(Tctd1.T, Tctd2.T, '.')
        plt.xlabel('CTD T at 10m')
        plt.ylabel('CTD T at 20m')
        dcpy.plots.line45()

        plt.subplot2grid((4,2), (2,1))
        plt.plot(Tchi.T1[chirange], Tchi.T2[chirange], '.')
        plt.xlabel('χ-pod T₁')
        plt.ylabel('χ-pod T₂')
        dcpy.plots.line45()

        plt.subplot2grid((4,2),(3,0))
        T12 = (Tctd1.T + Tctd2.T)/2
        Tchi12 = np.interp(mpl.dates.date2num(Tctd1.time),
                           mpl.dates.date2num(Tchi.time[chirange]),
                           (Tchi.T1[chirange] + Tchi.T2[chirange])/2)
        plt.plot(T12, Tchi12, '.')
        plt.xlabel('CTD (10m + 20m)/2')
        plt.ylabel('χ-pod (T₁ + T₂)/2')
        dcpy.plots.line45()
        plt.grid()
        plt.tight_layout()

    def PlotEstimate(self, varname, est, hax=None, filter_len=None):

        import matplotlib.pyplot as plt
        import numpy as np

        self.LoadChiEstimates()

        if est == 'best':
            est = self.best

        if hax is None:
            hax = plt.gca()

        try:
            time = self.chi[est]['time'].squeeze()
        except:
            time = self.time

        if varname == 'chi' or varname == 'χ':
            var = self.chi[est]['chi'][:].squeeze()
            titlestr = '$χ$'
            yscale = 'log'
            grdflag = True

        if varname == 'KT' or varname == 'Kt':
            # self.CalcKT()
            var = self.KT[est][:].squeeze()
            var[var < 0] = np.nan
            titlestr = '$K_T$'
            yscale = 'log'
            grdflag = True

        if varname == 'Jq':
            var = self.Jq[est]
            titlestr = '$J_q$'
            yscale = 'linear'
            grdflag = False

        if filter_len is not None:
            dt = np.diff(time[0:2])*86400
            filter_len /= dt
            import bottleneck as bn
            if varname == 'Jq':
                var = bn.move_mean(var, window=filter_len,
                                   min_count=filter_len/5)
            else:
                var = bn.move_median(var, window=filter_len,
                                     min_count=filter_len/5)

        # dtime = dcpy.util.datenum2datetime(time)
        hax.plot_date(time, var, '-', label=est)
        hax.set(yscale=yscale)
        # hax.set_xlim([dtime[0], dtime[-1]])
        plt.grid(grdflag, axis='y', which='major')

        hax.set_title(titlestr + ' ' + est + self.name)

    def CompareEstimates(self, varname, est1, est2, filter_len=None):
        import numpy as np
        import matplotlib.pyplot as plt
        import dcpy.plots

        # time = self.chi[est1]['time'][0:-1:10]
        if varname == 'chi':
            var1 = self.chi[est1]['chi'][:].squeeze()
            var2 = self.chi[est2]['chi'][:].squeeze()
            titlestr = 'χ'

        if varname == 'KT':
            self.CalcKT()
            var1 = self.KT[est1]
            var2 = self.KT[est2]
            titlestr = 'K_T'

        plt.subplot(2, 2, (1, 2))
        hax = plt.gca()
        self.PlotEstimate(varname, est1, hax, filter_len)
        self.PlotEstimate(varname, est2, hax, filter_len)
        hax.set_title(titlestr + ' | ' + self.name)

        plt.subplot(2, 2, 3)
        hax = plt.gca()
        lv1 = np.log10(var1)
        lv2 = np.log10(var2)
        plt.hist(lv1[np.isfinite(lv1)], bins=40,
                 alpha=0.5, normed=True, label=est1)
        plt.hist(lv2[np.isfinite(lv2)], bins=40,
                 alpha=0.5, normed=True, label=est2)
        hax.legend()

        try:
            mask12 = np.isnan(var1) | np.isnan(var2)
        except:
            # should only run when comparing
            # against sally's estimate.
            t1 = self.chi[est1]['time'].squeeze()
            t2 = self.chi[est2]['time'].squeeze()
            var1 = np.interp(t2, t1, var1.squeeze())
            mask12 = np.isnan(var1) | np.isnan(var2)

        var1 = var1[~mask12]
        var2 = var2[~mask12]

        plt.subplot(2, 2, 4)
        hax = plt.gca()
        hax.hexbin(np.log10(var1), np.log10(var2), cmap=plt.cm.YlOrRd)
        hax.set_xlabel(titlestr + '_' + est1)
        hax.set_ylabel(titlestr + '_' + est2)
        dcpy.plots.line45()
        # lims = [1e-10, 1e-4]
        # plt.xlim(lims); plt.ylim(lims)
