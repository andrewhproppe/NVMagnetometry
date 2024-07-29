import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import t
from src.modules.g2_funks import peak_area_ratio
from src.visualization.AP_figs_funcs import dress_fig, make_fig, set_font_size


class RMSE(object):
    """
    Calculates Root Mean Squared Error of a list of y arrays versus a target y array
    Inputs:
    y_test_set: list of test y arrays
    y_target: y array to which each test y array is compared
    """

    def __init__(self, y_test, y_true, one_true=False, normalize=False):
        self.avg = 0
        self.std = 0
        self.rmse = 0
        self.y_test = y_test
        self.y_true = y_true
        self.one_true = one_true
        self.normalize = normalize
        self.compute_RMSE(self.y_test, self.y_true, self.one_true)

    def compute_RMSE(self, y_test, y_true, one_true):
        rmse = np.zeros(len(y_test))
        for i in range(0, len(rmse)):
            if one_true:
                y1 = y_true[0]
            else:
                y1 = y_true[i]

            y2 = y_test[i]

            if self.normalize:
                y1 = y1 / max(y1)
                y2 = y2 / max(y2)

            rmse[i] = mean_squared_error(y1, y2, squared=False)
            # if one_true:
            #     rmse[i] = (mean_squared_error(y_true[0], y_test[i], squared=False))
            # else:
            #     rmse[i] = (mean_squared_error(y_true[i], y_test[i], squared=False))

        self.rmse = rmse
        self.avg = rmse.mean()
        self.std = rmse.std()


class ThetaError(object):
    """
    Calculates the average value, standard deviation, average error (in %), and error standard deviation for a set of fitted parameters versus true parameters
    Inputs:
    theta_test: list of test parameter arrays
    theta_true: true array of parameters
    """

    def __init__(self, theta_test, theta_true, one_true=False):
        self.avg = 0
        self.std = 0
        self.error = None
        self.error_avg = 0
        self.error_std = 0
        self.theta_test = np.array(theta_test)
        self.theta_true = np.array(theta_true)
        self.one_true = one_true
        self.compute()

    def compute(self, theta_test=None, theta_true=None, one_true=None):
        if theta_test == None:
            theta_test = self.theta_test
        if theta_true == None:
            theta_true = self.theta_true
        if one_true == None:
            one_true = self.one_true

        theta_test = np.array(theta_test)
        theta_true = np.array(theta_true)
        theta_error = np.zeros(shape=theta_test.shape)

        for i in range(0, len(theta_error)):
            # theta_error[i] = np.abs((1 - (theta_test[i] / theta_true[i]))*100)
            if one_true:
                theta_error[i] = np.abs(theta_test[i] - theta_true[0]) * 100
            else:
                theta_error[i] = np.abs(theta_test[i] - theta_true[i]) * 100

        # for i, theta_test in enumerate(theta_test):
        #     theta_error[i] = np.abs((1 - (theta_test / theta_true))*100)

        self.theta_true = theta_true
        self.theta_test = theta_test
        self.avg = theta_test.mean(axis=0)
        self.std = theta_test.std(axis=0)
        self.error = theta_error
        self.error_avg = theta_error.mean(axis=0)
        self.error_std = theta_error.std(axis=0)

    def plot_histograms(
        self, nbins=None, var_names=None, fig_ax=None, vars=None, bin_ranges=None
    ):
        if nbins == None:
            nbins = 20

        if var_names == None:
            var_names = ["w1", "w2", "w3", "w4", "w5", "w6", "w7"]

        if vars == None:
            try:
                vars = np.arange(len(self.theta_test[0]))
            except:
                vars = [0]

        if fig_ax == None:
            fig, ax = plt.subplots(
                nrows=len(vars), dpi=150, figsize=(4, 4), squeeze=False
            )
            ax = ax.flatten()
        else:
            fig, ax = fig_ax
            ax = ax.flatten()

        for ctr, i in enumerate(vars):
            if len(vars) == 1:
                y = self.theta_test[:]
            else:
                y = self.theta_test[:, i]
            if not bin_ranges == None:
                x = np.linspace(bin_ranges[i][0], bin_ranges[i][1], nbins)
                ax[ctr].hist(y, bins=x)
            else:
                ax[ctr].hist(y, bins=nbins)
            ax[ctr].set_xlabel(var_names[i])

    def plot_true_vs_pred(
        self,
        vars=None,
        marker="s",
        var_names=None,
        label=None,
        color=None,
        fig_ax=None,
        r2=True,
        figsize=(4, 4),
    ):
        if var_names == None:
            var_names = ["w1", "w2", "w3", "w4", "w5", "w6", "w7"]

        if vars == None:
            try:
                vars = np.arange(len(self.theta_test[0]))
            except:
                vars = [0]

        if fig_ax == None:
            fig, ax = plt.subplots(
                nrows=len(vars), dpi=150, figsize=figsize, squeeze=False
            )
            ax = ax.flatten()
        else:
            fig, ax = fig_ax
            ax = ax.flatten()

        for ctr, i in enumerate(vars):
            if len(vars) == 1:
                y_test = self.theta_test[:]
                y_true = self.theta_true[:]
            else:
                y_test = self.theta_test[:, i]
                y_true = self.theta_true[:, i]

            ax[ctr].scatter(
                y_true, y_test, s=2, marker=marker, color=color, label=label
            )
            ax[ctr].set_xlabel("True " + var_names[i])
            ax[ctr].set_ylabel("Predicted " + var_names[i])
            if r2:
                r2_val = r2_score(y_true=y_true, y_pred=y_test)
                # plt.text(x=min(y_true), y=max(y_true)*0.8, s='R2 = %.4f' % r2_val)
                plt.text(x=0.5, y=0, s="R2 = %.4f" % r2_val)
            dress_fig(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], legend=True)


class PeakAreaRatio(object):
    """
    Calculates the peak-area-ratio for a list of fitted pulsed g2 functions
    Inputs:
    g2_test: list of test g2s
    g2_target: list of true g2s
    g2_t: time array of g2 photon lag times (in µs)
    window_size: time window over which to integrate each peak (in µs)
    rep_rate: repetition rate of the laser pulse used in the experiment (in µs)
    """

    def __init__(self, g2_test, g2_true, g2_t, window_size, rep_rate, one_true=False):
        self.test = None
        self.true = None
        self.error = None
        self.error_avg = None
        self.error_std = None
        self.g2_test = g2_test
        self.g2_true = g2_true
        self.g2_t = g2_t
        self.window_size = window_size
        self.rep_rate = rep_rate
        self.one_true = one_true
        self.compute()

    def compute(
        self,
        g2_test=None,
        g2_true=None,
        g2_t=None,
        window_size=None,
        rep_rate=None,
        one_true=None,
    ):
        if g2_test is None:
            g2_test = self.g2_test

        if g2_true is None:
            g2_true = self.g2_true

        if g2_t is None:
            g2_t = self.g2_t

        if window_size is None:
            window_size = self.window_size

        if rep_rate is None:
            rep_rate = self.rep_rate

        if one_true is None:
            one_true = self.one_true

        self.test = np.zeros(len(g2_test))
        self.true = np.zeros(len(g2_true))
        par_error = np.zeros(shape=self.test.shape)

        # Compute test peak-area-ratios
        for i, g2 in enumerate(g2_test):
            self.test[i] = peak_area_ratio(
                g2=g2, g2_t=g2_t, window_size=window_size, rep_rate=rep_rate
            )

        # Compute true peak-area-ratios
        for i, g2 in enumerate(g2_true):
            self.true[i] = peak_area_ratio(
                g2=g2, g2_t=g2_t, window_size=window_size, rep_rate=rep_rate
            )

        for i in range(0, len(par_error)):
            if one_true:
                par_error[i] = np.abs(self.test[i] - self.true[0]) * 100
            else:
                par_error[i] = np.abs(self.test[i] - self.true[i]) * 100

        self.error = par_error
        self.error_avg = par_error.mean(axis=0)
        self.error_std = par_error.std(axis=0)

    def plot_true_vs_pred(
        self,
        marker="s",
        var_names=None,
        fill=True,
        label=None,
        color=None,
        fig_ax=None,
        r2=True,
        figsize=(4, 4),
        scatter_alpha=0.8,
        fill_alpha=0.1,
        r2_text_pos=0.75,
        fontsize=7,
    ):
        set_font_size(fontsize)

        if fig_ax == None:
            fig, ax = make_fig(figsize[0], figsize[1])
        else:
            fig = fig_ax

        self.true, self.test = zip(*sorted(zip(self.true, self.test)))
        self.true = np.array(self.true)
        self.test = np.array(self.test)

        plt.scatter(
            self.true,
            self.test,
            s=2,
            marker=marker,
            color=color,
            linewidths=0,
            label=label,
            alpha=scatter_alpha,
        )
        plt.xlabel("True Ratio")
        plt.ylabel("Predicted Ratio")

        if r2:
            x = np.arange(0, 1, 0.01)
            y = x
            plt.plot(x, y, color="k", lw=1)
            r2_val = r2_score(y_true=self.true, y_pred=self.test)
            plt.text(x=r2_text_pos, y=0, s="$\itr^{2}$ = %.2f" % r2_val)
            self.r2 = r2_val
            # plt.legend(loc='upper left')

        # Old
        # if r2:
        #     res = stats.linregress(self.true, self.test)
        #     tinv = lambda p, df: abs(t.ppf(p/2, df))
        #     ts = tinv(0.05, len(self.true) - 2)
        #     linear = res.slope*self.true+res.intercept
        #     linear_ub = (res.slope+ts*res.stderr)*self.true+(res.intercept+ts*res.stderr)
        #     linear_lb = (res.slope-ts*res.stderr)*self.true+(res.intercept-ts*res.stderr)
        #     plt.plot(self.true, linear, color='k', lw=1)
        #     if fill:
        #         plt.fill_between(self.true, linear_ub, linear_lb, color=color, alpha=fill_alpha)
        #     plt.text(x=r2_text_pos, y=0, s='$\itr^{2}$ = %.2f' % res.rvalue**2)
        #     self.r2 = res.rvalue**2
        #     # plt.legend(loc='upper left')

        dress_fig(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], legend=False, tight=False)


class PeakHeightRatio(object):
    """
    Calculates the peak-height-ratio for a list of fitted pulsed g2 functions
    Inputs:
    g2_test: list of test g2s
    g2_target: list of true g2s
    g2_t: time array of g2 photon lag times (in µs)
    """

    def __init__(self, g2_test, g2_true, g2_t, one_true=False):
        self.test = None
        self.true = None
        self.error = None
        self.error_avg = None
        self.error_std = None
        self.g2_test = g2_test
        self.g2_true = g2_true
        self.g2_t = g2_t
        self.one_true = one_true
        self.compute()

    def compute(self, g2_test=None, g2_true=None, g2_t=None, one_true=None):
        if g2_test is None:
            g2_test = self.g2_test

        if g2_true is None:
            g2_true = self.g2_true

        if g2_t is None:
            g2_t = self.g2_t

        if one_true is None:
            one_true = self.one_true

        self.test = np.zeros(len(g2_test))
        self.true = np.zeros(len(g2_true))
        h_error = np.zeros(shape=self.test.shape)

        # Compute test peak-area-heights
        for i, g2 in enumerate(g2_test):
            self.test[i] = g2[0] / max(g2)

        # Compute true peak-area-ratios
        for i, g2 in enumerate(g2_true):
            self.true[i] = g2[0] / max(g2)

        for i in range(0, len(h_error)):
            if one_true:
                h_error[i] = np.abs(self.test[i] - self.true[0]) * 100
            else:
                h_error[i] = np.abs(self.test[i] - self.true[i]) * 100

        self.error = h_error
        self.error_avg = h_error.mean(axis=0)
        self.error_std = h_error.std(axis=0)

    def plot_true_vs_pred(
        self,
        marker="s",
        var_names=None,
        fill=True,
        label=None,
        color=None,
        fig_ax=None,
        r2=True,
        figsize=(4, 4),
        scatter_alpha=0.8,
        fill_alpha=0.1,
        r2_text_pos=0.75,
        fontsize=7,
    ):
        set_font_size(fontsize)

        if fig_ax == None:
            fig, ax = make_fig(figsize[0], figsize[1])
        else:
            fig = fig_ax

        self.true, self.test = zip(*sorted(zip(self.true, self.test)))
        self.true = np.array(self.true)
        self.test = np.array(self.test)

        plt.scatter(
            self.true,
            self.test,
            s=2,
            marker=marker,
            color=color,
            linewidths=0,
            label=label,
            alpha=scatter_alpha,
        )
        plt.xlabel("True Ratio")
        plt.ylabel("Predicted Ratio")
        if r2:
            res = stats.linregress(self.true, self.test)
            tinv = lambda p, df: abs(t.ppf(p / 2, df))
            ts = tinv(0.05, len(self.true) - 2)
            linear = res.slope * self.true + res.intercept
            linear_ub = (res.slope + ts * res.stderr) * self.true + (
                res.intercept + ts * res.stderr
            )
            linear_lb = (res.slope - ts * res.stderr) * self.true + (
                res.intercept - ts * res.stderr
            )
            plt.plot(self.true, linear, color="k", lw=1)
            if fill:
                plt.fill_between(
                    self.true, linear_ub, linear_lb, color=color, alpha=fill_alpha
                )
            plt.text(x=r2_text_pos, y=0, s="$\itr^{2}$ = %.2f" % res.rvalue**2)
            self.r2 = res.rvalue**2
            # plt.legend(loc='upper left')
        dress_fig(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], legend=False, tight=False)
