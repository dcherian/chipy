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
        if chifile == 'Turb.mat':
            self.chifile = basedir + unit + '/proc/' + chifile
        else:
            self.chifile = basedir + unit + '/proc/combined/' + chifile

        self.depth = depth
        self.χestimates = []
        self.time = []
        self.best = best
        self.dt = []
        self.Tchi = []

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
        self.LoadTzi()
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

        import hdf5storage as hs

        mat = hs.loadmat(self.procdir + '/temp.mat',
                         struct_as_record=False, squeeze_me=True)

        self.Tchi = mat['T']
        self.Tchi['time'] = self.Tchi['time'] - 367

    def LoadTzi(self):
        ''' Load internal stratification estimate '''
        import hdf5storage as hs

        mat = hs.loadmat(self.procdir + '../input/dTdz_i.mat',
                         struct_as_record=False, squeeze_me=True)
        try:
            self.Tzi = mat['Tz_i']
            self.Tzi['time'] -= 367
        except:
            pass

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

            if type(var[e1]) == np.void:
                self.χestimates.append(ff)
                var[ff]['chi'] = np.nanmean(
                    [var[e1]['chi'], var[e2]['chi']], axis=0)
                var[ff]['T'] = np.nanmean(
                    [var[e1]['T'], var[e2]['T']], axis=0)
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

    def FilterEstimate(self, kind, time, var, order=1,
                       filter_len=None, decimate=False):
        import numpy as np

        dt = np.diff(time[0:2])*86400
        if filter_len is not None:
            if kind == 'bandpass':
                import dcpy.ts
                from dcpy.util import MovingAverage
                filter_len = np.array(filter_len)
                # runnning average to remove gaps
                var = MovingAverage(var, filter_len.min()/2/dt, decimate=False)
                var = dcpy.ts.BandPassButter(var, 1/filter_len, dt,
                                             order=order, num_discard=5)
            else:
                filter_len = np.int(np.floor(filter_len/dt))
                if np.mod(filter_len, 2) == 0:
                    filter_len = filter_len - 1

                import bottleneck as bn
                time = bn.move_mean(time, window=filter_len,
                                    min_count=1)
                if kind == 'mean' or kind == 'Jq':
                    var = bn.move_mean(var, window=filter_len,
                                       min_count=1)
                else:
                    var = bn.move_median(var, window=filter_len,
                                         min_count=1)

                if decimate is True:
                    # subsample
                    L = filter_len
                    Lb2 = np.int(np.floor(filter_len/2))
                    time = time[Lb2+1::L]
                    var = var[Lb2+1::L]

        return time, var

    def PlotEstimate(self, varname, est, hax=None, filt=None,
                     filter_len=None, tind=None):

        import matplotlib.pyplot as plt

        self.LoadChiEstimates()

        if hax is None:
            hax = plt.gca()

        try:
            time = self.chi[est]['time'].squeeze()
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
                                        filter_len=filter_len, decimate=False)

        if varname == 'Jq':
            import numpy as np
            var[np.isnan(var)] = 0

        hax.plot(time, var, label=est, linewidth=1)
        hax.xaxis_date()
        hax.set_ylabel(titlestr)
        hax.set(yscale=yscale)
        plt.grid(grdflag, axis='y', which='major')

        hax.set_title(titlestr + ' ' + est + self.name)

    def PlotSpectrum(self, varname, est='best', nsmooth=5,
                     filter_len=None, SubsetLength=None,
                     ticks=None, ax=None, norm=False, **kwargs):

        import numpy as np
        from dcpy.ts import SpectralDensity
        import matplotlib.pyplot as plt

        var, titlestr, _, _ = self.ChooseVariable(varname, est)

        t, χ = self.FilterEstimate('mean', self.time, var,
                                   filter_len=filter_len,
                                   decimate=True)
        dt = np.nanmean(np.diff(t)*86400)

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
            addstr = '/ $\int$ PSD'

        from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
        if ax is None:
            fig = plt.figure()
            ax = SubplotHost(fig, 1, 1, 1)
            fig.add_subplot(ax)

        hdl = ax.loglog(1/f, S, label=str(self.depth)+' m')
        if len(conf) > 2:
            ax.fill_between(1/f, conf[:, 0], conf[:, 1],
                            color=hdl[0].get_color(), alpha=0.3)

        ax.set_ylabel('PSD( ' + titlestr + ' )' + addstr)
        ax.set_xlim([np.min(1/f), np.max(1/f)])

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

        tickstr = [str(np.round(xx/norm, 2)) for xx in ticks]
        ax.set_xticklabels(tickstr)
        ax.set_xlabel('Period (' + tstr + ')')
        ax.legend()
        ax.grid(True)

        if not ax.xaxis_inverted():
            ax.invert_xaxis()

        return ax

    def CompareEstimates(self, varname, est1, est2, filter_len=None):
        import numpy as np
        import matplotlib.pyplot as plt
        import dcpy.plots

        # time = self.chi[est1]['time'][0:-1:10]
        if varname == 'chi' or varname == 'χ':
            var1 = self.chi[est1]['chi'][:].squeeze()
            var2 = self.chi[est2]['chi'][:].squeeze()
            titlestr = 'χ'

        if varname == 'KT' or varname == 'Kt':
            var1 = self.KT[est1]
            var2 = self.KT[est2]
            titlestr = 'K_T'

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
                 alpha=0.5, normed=True, label=est1)
        plt.hist(lv2[np.isfinite(lv2)], bins=40,
                 alpha=0.5, normed=True, label=est2)
        plt.xlabel(titlestr)
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
        hax.set_xlabel('${' + titlestr + '}^{' + est1 + '}$')
        hax.set_ylabel('${' + titlestr + '}^{' + est2 + '}$')
        dcpy.plots.line45()

        plt.tight_layout()
        # lims = [1e-10, 1e-4]
        # plt.xlim(lims); plt.ylim(lims)

    def Summarize(self, est='best', filter_len=None, tind=None):

        import matplotlib.pyplot as plt
        import numpy as np

        if est == 'best':
            est = self.best

        time = self.chi[est]['time']

        if tind is None:
            tind = range(len(time))

        plt.figure(figsize=[6, 8.5])
        ax1 = plt.subplot(7, 1, 1)
        ax1.plot_date(time[tind], self.chi[est]['N2'][tind],
                      '-', linewidth=0.5)
        ax1.set_ylabel('$N^2$')

        ax2 = plt.subplot(7, 1, 2, sharex=ax1)
        ax2.plot_date(time[tind],
                      self.chi[est]['dTdz'][tind],
                      '-', linewidth=0.5)

        ind0 = np.where(self.Tzi['time'][0, 0]
                        > time[tind][0])[0][0]
        ind1 = np.where(self.Tzi['time'][0, 0]
                        < time[tind][-2])[0][-1]
        ax2.plot_date(self.Tzi['time'][0, 0][ind0:ind1],
                      self.Tzi['Tz1'][0, 0][ind0:ind1], '-',
                      linewidth=0.5)
        plt.axhline(0, color='k', zorder=-1)
        plt.axhline(5e-4, color='k', zorder=-1)
        plt.axhline(-5e-4, color='k', zorder=-1)
        ax2.set_ylabel('dT/dz')

        ax3 = plt.subplot(7, 1, 3, sharex=ax1)
        if self.Tchi:
            if filter_len is None:
                # at minimum, average differentiator
                # to same time resolution as χ estimate
                fl = (time[2]-time[1])*86400
            else:
                fl = filter_len

            fl = None

            time, var = self.FilterEstimate(varname='Jq',
                                            time=self.Tchi['time'][0, 0],
                                            var=self.Tchi['T1Pt'][0, 0],
                                            filter_len=fl)
            ind0 = np.where(time > time[tind][0])[0][0]
            ind1 = np.where(time < time[tind][-2])[0][-1]

            ax3.plot(time[ind0:ind1], var[ind0:ind1], '-', linewidth=0.5)

            # time, var = self.FilterEstimate(varname='variance',
            #                                 time=self.Tchi['time'][0, 0],
            #                                 var=self.Tchi['T2Pt'][0, 0],
            #                                 filter_len=fl)
            # ax3.plot(time[ind0:ind1], var[ind0:ind1], '-', linewidth=0.5)
            ax3.set_ylabel('var(T)')

        ax4 = plt.subplot(7, 1, 4, sharex=ax1)
        self.PlotEstimate('T', est=est,
                          filter_len=None, tind=tind,
                          hax=ax4)
        ax4.set_title('')

        ax5 = plt.subplot(7, 1, 5, sharex=ax1)
        self.PlotEstimate('chi', est=est,
                          filter_len=filter_len, tind=tind,
                          hax=ax5)
        ax5.set_title('')

        ax6 = plt.subplot(7, 1, 6, sharex=ax1)
        self.PlotEstimate('KT', est=est,
                          filter_len=filter_len, tind=tind,
                          hax=ax6)
        ax6.set_title('')

        ax7 = plt.subplot(7, 1, 7, sharex=ax1)
        self.PlotEstimate('Jq', est=est,
                          filter_len=filter_len, tind=tind,
                          hax=ax7)
        ax7.set_title('')

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.setp(ax5.get_xticklabels(), visible=False)
        plt.gcf().autofmt_xdate()
        ax1.set_xlim([np.nanmin(time[tind]),
                      np.nanmax(time[tind])])

        plt.tight_layout()
        plt.show()

    def SeasonalSummary(self, ax=None, idx: int=0,
                        filter_len: int=86400):

        def ReturnSeason(time, var, season):
            ''' Given a season, return data only for the months in that season
            season can be one of SW, NE, SW->NE or NE->SW
            '''
            from dcpy.util import datenum2datetime
            mask = np.isnan(time)
            time = time[~mask]
            var = var[~mask]

            dates = datenum2datetime(time)
            months = [d.month for d in dates]

            seasonMonths = {'SW':  [5, 6, 7, 8, 9],
                            'SW→NE': [10, 11],
                            'NE':  [12, 1, 2],
                            'NE→SW': [3, 4]}

            mask = np.asarray([m in seasonMonths[season] for m in months])

            return time[mask], var[mask]

        import matplotlib.pyplot as plt
        import numpy as np

        seasons = ['NE', 'NE→SW', 'SW', 'SW→NE']
        cpodcolor = ['r', 'b']
        handles = []

        # positioning parameters
        n = 1.2
        x0 = 0
        pos = []
        label = []

        if ax is None:
            plt.figure(figsize=[6.5, 4.5])
            ax = plt.gca()
            ax.set_title(self.name)

        varname = 'KT'
        var = self.KT[self.best]
        time = self.time
        label.append(self.name)

        time, var = self.FilterEstimate(kind=varname,
                                        time=time, var=var,
                                        filter_len=filter_len)

        for sidx, ss in enumerate(seasons):
            _, varex = ReturnSeason(time, var, ss)
            style = {'color': cpodcolor[idx]}
            meanstyle = {'color': cpodcolor[idx],
                         'marker': '.'}
            pos.append(x0 + n*(sidx+1)+(idx-0.5)/3)
            hdl = ax.boxplot(np.log10(varex[~np.isnan(varex)]),
                             positions=[pos[-1]],
                             showmeans=True,
                             boxprops=style, medianprops=style,
                             meanprops=meanstyle)
            handles.append(hdl)

        pos = np.sort(np.array(pos))
        ax.set_xticks(pos)

        ax.set_xticklabels(seasons)
        ax.set_xlim([pos[0]-0.5-x0, pos[-1]+0.5])
        ax.set_ylabel('$K_T$')
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

        if varname == 'chi' or varname == 'χ':
            var = self.chi[est]['chi'][:].squeeze()
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
            var = self.chi[est]['T'][:].squeeze()
            titlestr = '$T$'
            yscale = 'linear'
            grdflag = False

        return [var, titlestr, yscale, grdflag]
