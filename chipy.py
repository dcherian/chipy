class chipod:
    """ Base class for chipod instruments.
        Depends heavily on output from chipod_gust routines.
        Represents chipod + ancillary data associated with that
        particular chipod
    """

    def __init__(self, basedir, unit, chifile=''):
        self.basedir = basedir
        self.unit = unit
        self.name = unit + ' | ' + chifile

        # setup dirs for chipod_gust
        self.inputdir = basedir + unit + '/input/'
        self.procdir = basedir + unit + '/proc/'
        self.chifile = basedir + unit + '/proc/combined/' + chifile

        self.χestimates = []

        # import os
        # this lets me import sally's processed output
        # if not os.path.isdir(self.chidir):
        #    self.chidir = basedir

        # nearby mooring instruments
        self.ctd = dict([])
        self.adcp = dict([])

        # derived quantities
        self.chi = dict([])
        self.KT = dict([])

    def LoadCTD(self):
        ''' Loads data from proc/T_m.mat '''
        import hdf5storage as hs
        import dcpy.util

        mat = hs.loadmat(self.procdir + '/T_m.mat',
                         struct_as_record=False, squeeze_me=True)
        self.ctd1 = mat['T1']; self.ctd2 = mat['T2'];
        self.ctd1.time = dcpy.util.datenum2datetime(self.ctd1.time)
        self.ctd2.time = dcpy.util.datenum2datetime(self.ctd2.time)

    def LoadT1T2(self):
        ''' Loads data from internal χ-pod sensors '''
        import scipy.io as spio

        mat = spio.loadmat(self.procdir + '/temp.mat',
                           struct_as_record=False, squeeze_me=True)
        self.Tchi = mat['T']
        self.Tchi.time = dcpy.util.datenum2datetime(self.Tchi.time)

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

        # average together similar estimates
        if 'mm1' in self.chi and 'mm2' in self.chi:
            self.chi['mm'] = dict()
            self.chi['mm']['chi'] = (self.chi['mm1']['chi']
                                     + self.chi['mm2']['chi'])/2

        if 'mi11' in self.chi and 'mi22' in self.chi:
            self.chi['mi'] = dict()
            self.chi['mi']['chi'] = (self.chi['mi11']['chi']
                                     + self.chi['mi22']['chi'])/2

        if 'pm1' in self.chi and 'pm2' in self.chi:
            self.chi['pm'] = dict()
            self.chi['pm']['chi'] = (self.chi['pm1']['chi']
                                     + self.chi['pm2']['chi'])/2

        if 'pi11' in self.chi and 'pi22' in self.chi:
            self.chi['pi'] = dict()
            self.chi['pi']['chi'] = (self.chi['pi11']['chi']
                                     + self.chi['pi22']['chi'])/2

    def LoadSallyChiEstimate(self, fname, estname):
        ''' fname - the mat file you want to read from.
            estname - what you want to name the estimate.

            Output saved as self.chi[estname] '''

        import os
        import glob
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

    def CalcKT(self):
        self.LoadChiEstimates()

        for est in self.χestimates:
            chi = self.chi[est]['chi'][:]
            dTdz = self.chi[est]['dTdz'][:]
            KT = (0.5*chi)/dTdz**2
            self.KT[est] = KT

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
        import dcpy.util
        import numpy as np

        self.LoadChiEstimates()

        if hax is None:
            hax = plt.gca()

        time = self.chi[est]['time'][:].squeeze()

        if varname == 'chi':
            var = self.chi[est]['chi'][:].squeeze()
            titlestr = 'χ'

        if varname == 'KT':
            self.CalcKT()
            var = self.KT[est][:].squeeze()
            var[var < 0] = np.nan
            titlestr = 'K_T'

        if filter_len is not None:
            import bottleneck as bn
            var = bn.move_median(var, window=filter_len,
                                 min_count=filter_len/5)

        hax.plot_date(dcpy.util.datenum2datetime(time), var, '-', label=est)
        hax.set(yscale='log')

        hax.set_title(titlestr + ' ' + est + self.name)

    def CompareEstimates(self, varname, est1, est2, filter_len=None):
        import numpy as np
        import matplotlib.pyplot as plt
        import dcpy.plots

        # time = self.chi[est1]['time'][0:-1:10]
        if varname == 'chi':
            var1 = self.chi[est1]['chi'][:].squeeze()
            var1[var1 < 0] = np.nan
            var2 = self.chi[est2]['chi'][:].squeeze()
            var2[var2 < 0] = np.nan
            titlestr = 'χ'

        if varname == 'KT':
            self.CalcKT()
            var1 = self.KT[est1]
            var2 = self.KT[est2]
            titlestr = 'K_T'

        plt.subplot(3, 1, 1)
        hax = plt.gca()
        self.PlotEstimate(varname, est1, hax, filter_len)
        self.PlotEstimate(varname, est2, hax, filter_len)
        hax.set_title(titlestr + ' | ' + self.name)

        plt.subplot(3, 1, 2)
        hax = plt.gca()
        lv1 = np.log10(var1)
        lv2 = np.log10(var2)
        plt.hist(lv1[np.isfinite(lv1)], bins=40,
                 alpha=0.5, normed=True, label=est1)
        plt.hist(lv2[np.isfinite(lv2)], bins=40,
                 alpha=0.5, normed=True, label=est2)
        hax.legend()

        mask12 = np.isnan(var1) | np.isnan(var2)
        var1 = var1[~mask12]
        var2 = var2[~mask12]

        plt.subplot(3, 1, 3)
        hax = plt.gca()
        hax.hexbin(np.log10(var1), np.log10(var2), cmap=plt.cm.YlOrRd)
        hax.set_xlabel(titlestr + '_' + est1)
        hax.set_ylabel(titlestr + '_' + est2)
        dcpy.plots.line45()
        # lims = [1e-10, 1e-4]
        # plt.xlim(lims); plt.ylim(lims)
