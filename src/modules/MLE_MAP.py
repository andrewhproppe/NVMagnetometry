import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.signal import correlate
from scipy.special import erf as erf
from math import ceil


class MLE_MultiObjective(object):
    """
    Object that is filled with parameter arrays, indices for best optimal parameters, and fits with optimal parameters from MLE estimation
    """
    def __init__(self, x, y, nruns, nvars, fit_function, obj_functions, **kw):

        self.x = x
        self.y = y
        self.nruns = nruns
        self.nvars = nvars
        self.fit_function = fit_function
        self.obj_functions = obj_functions
        self.kwargs = kw
    
        self.idxP = []
        self.idxL = []
        self.idxPL = []

        self.thetaFinalP = []
        self.thetaFinalL = []
        self.thetaFinalPL = []

        self.theta_P = []
        self.theta_L = []
        self.theta_PL = []

        self.fit_P = []
        self.fit_L = []
        self.fit_PL = []

        self.LossFinalP = []
        self.LossFinalL = []
        self.LossFinalPL = []

        self.ElapsedTime = []

        self.find_MLE(x=self.x, y=self.y, nruns=self.nruns, nvars=self.nvars, fit_function=self.fit_function, obj_functions=self.obj_functions, **self.kwargs)

    def find_MLE(self, x, y, nruns, nvars, fit_function, obj_functions, printing=True, **kwargs):
        """
        Written by Cristian L. Cortes, Argonne National Lab, 2020
        Modified by Andrew H. Proppe, Massachusetts Institute of Technology, 2021

        Inputs:
        x = x data points, numpy array
        y = y data points, numpy array
        nruns = number of optimizations with (optionally) different initial guesses
        nvars = number of parameters used in the fit equation
        guess = optional input array of initial guess for optimization
        randomizer = value between 0 and 1 that blends input guess with random guess (0 for 100% input guess, 1 for 100% random guess)
                     or if bounds are given, use 'bounded' to generate random guesses between lower and upper bounds
        bounds = tuple of lower and upper bounds (e.g. ((0,1),(10,100))) that work with certain scipy minimization methods
        random_range = decade of range that is used for random guesses (e.g. random guess between 0 and 10^(random_range))
        """
        np.seterr(divide='ignore', invalid='ignore', over='ignore')
        t = time.time()

        opts = {'maxiter': 10000,
                'disp': True
                #         'maxfun'  : 10000,
                #         'disp' : True,
                # 'full_output': True,
                # 'ftol': 1e-15,
                #         'ftol' : 1e-14,
                # 'eps': 1e-15
                                }  # default value.

        if 'guess' in kwargs:
            input_guess = kwargs['guess']

        if 'randomizer' in kwargs:
            randomizer  = kwargs['randomizer']

        if 'random_range' in kwargs:
            random_range  = kwargs['random_range']

        if 'bounds' in kwargs:
            bounds = kwargs['bounds']
            lb = np.array(bounds[0])
            ub = np.array(bounds[1])
            bds = np.array((lb, ub)).T
            bds = tuple(map(tuple, bds))

        nruns = nruns
        nvars = nvars  # number of fitting parameters

        thetaFinalP = np.zeros((nruns, nvars))
        thetaFinalL = np.zeros((nruns, nvars))
        thetaFinalPL = np.zeros((nruns, nvars))

        LossFinalP = np.zeros(nruns)
        LossFinalL = np.zeros(nruns)
        LossFinalPL = np.zeros(nruns)

        for k in tqdm(range(nruns), position=0, leave=True, desc='Finding MLE...', disable=not printing):  # perform optimization over multiple random initial conditions

            if 'guess' in kwargs:
                guess = np.array(input_guess)
            else:
                if 'random_range' in kwargs:
                    guess = np.random.uniform(0, 10**random_range, nvars)  # randomize initial guesses
                else:
                    guess = np.random.uniform(0, 100, nvars) # randomize initial guesses

            if 'randomizer' in kwargs and 'guess' in kwargs: # Use randomizer on input guess
                if randomizer == 'bounded' and 'bounds' in kwargs:
                    guess = np.random.uniform(0.01, 0.99, nvars) * (ub - lb) + lb
                else:
                    if 'random_range' in kwargs:
                        random_guess = np.random.uniform(0, 10**random_range, nvars) # Specify decade for range of random array
                    else:
                        random_guess = np.random.uniform(0, 1, nvars)
                    new_guess = random_guess * randomizer + guess * (1 - randomizer)
                    guess = new_guess

            if 'bounds' in kwargs:
                if 'Poisson' in obj_functions:
                    ResultP1 = minimize(lossP, guess, args=(x, y, fit_function), method='Powell', options=opts, bounds=bds)  # POISSON REGRESSION
                if 'Gaussian' in obj_functions:
                    ResultL1 = minimize(lossL, guess, args=(x, y, fit_function), method='Powell', options=opts, bounds=bds)  # LEAST SQUARES REGRESSION
                # ResultPL1 = minimize(lossPL, guess, args=(x, y, function), method='Powell', options=opts, bounds=bds) # HYBRID REGRESSION

            else:
                if 'Poisson' in obj_functions:
                    ResultP1 = minimize(lossP, guess, args=(x, y, fit_function), method='Powell', options=opts)  # POISSON REGRESSION
                if 'Gaussian' in obj_functions:
                    ResultL1 = minimize(lossL, guess, args=(x, y, fit_function), method='Powell', options=opts)  # LEAST SQUARES REGRESSION
                # ResultPL1 = minimize(lossPL, guess, args=(x, y, function), method='Powell', options=opts) # HYBRID REGRESSION

            if 'Poisson' in obj_functions:
                thetaFinalP[k, :] = ResultP1.x
                LossFinalP[k] = ResultP1.fun
                idxP = np.argmin(LossFinalP)
                mle_fit_P = function_clean(thetaFinalP[idxP, :], x, fit_function)  # Poisson
                theta_mleP = thetaFinalP[idxP, :]  # best parameters from mle (Poisson)
                self.idxP = idxP
                self.thetaFinalP = thetaFinalP
                self.theta_P = theta_mleP
                self.fit_P = mle_fit_P
                self.LossFinalP = LossFinalP

            if 'Gaussian' in obj_functions:
                thetaFinalL[k, :] = ResultL1.x
                LossFinalL[k] = ResultL1.fun
                idxL = np.argmin(LossFinalL)
                mle_fit_L = function_clean(thetaFinalL[idxL, :], x, fit_function)  # Gaussian
                theta_mleL = thetaFinalL[idxL, :]  # best parameters from mle (Gaussian)
                self.idxL = idxL
                self.thetaFinalL = thetaFinalL
                self.theta_L = theta_mleL
                self.fit_L = mle_fit_L
                self.LossFinalL = LossFinalL

            if 'Hybrid' in obj_functions:
                idxPL = np.argmin(LossFinalPL)
                mle_fit_PL = function_clean(thetaFinalPL[idxPL, :], x, fit_function)  # Hybrid
                theta_mlePL = thetaFinalL[idxPL, :]  # best parameters from mle (Hybrid)
                self.idxPL = idxPL
                self.thetaFinalPL = thetaFinalPL
                self.theta_PL = theta_mlePL
                self.fit_PL = mle_fit_PL
                self.LossFinalPL = LossFinalPL
                self.ElapsedTime = time.time() - t

        return self

class MLE_old(object):
    """
    Object that is filled with parameter arrays, indices for best optimal parameters, and fits with optimal parameters from MLE estimation
    """

    def __init__(self, nruns, nvars, obj_function, args=None, **kw):

        self.x = args[0]
        self.y = args[1]
        self.fit_function = args[2]
        self.nruns = nruns
        self.nvars = nvars
        self.obj_function = obj_function
        self.args = args
        self.kwargs = kw

        self.idx = []
        self.thetaFinal = []
        self.theta = []
        self.fit = []
        self.LossFinal = []
        self.ElapsedTime = []

        self.find_MLE(nruns=self.nruns, nvars=self.nvars, obj_function=self.obj_function, **self.kwargs)

    def find_MLE(self, nruns, nvars, obj_function, printing=True, display=True, **kwargs):
        """
        Written by Cristian L. Cortes, Argonne National Lab, 2020
        Modified by Andrew H. Proppe, Massachusetts Institute of Technology, 2021

        Inputs:
        x = x data points, numpy array
        y = y data points, numpy array
        nruns = number of optimizations with (optionally) different initial guesses
        nvars = number of parameters used in the fit equation
        guess = optional input array of initial guess for optimization
        randomizer = value between 0 and 1 that blends input guess with random guess (0 for 100% input guess, 1 for 100% random guess)
                     or if bounds are given, use 'bounded' to generate random guesses between lower and upper bounds
        bounds = tuple of lower and upper bounds (e.g. ((0,1),(10,100))) that work with certain scipy minimization methods
        random_range = decade of range that is used for random guesses (e.g. random guess between 0 and 10^(random_range))
        """
        np.seterr(divide='ignore', invalid='ignore', over='ignore')
        t = time.time()

        opts = {'maxiter': 10000,
                'disp': display,
                #         'maxfun'  : 10000,
                #         'disp' : True,
                # 'full_output': True,
                # 'ftol': 1e-15,
                #         'ftol' : 1e-14,
                # 'eps': 1e-15
                }  # default value.

        if 'guess' in kwargs:
            input_guess = kwargs['guess']

        if 'randomizer' in kwargs:
            randomizer = kwargs['randomizer']

        if 'random_range' in kwargs:
            random_range = kwargs['random_range']

        if 'bounds' in kwargs:
            bounds = kwargs['bounds']
            lb = np.array(bounds[0])
            ub = np.array(bounds[1])
            bds = np.array((lb, ub)).T
            bds = tuple(map(tuple, bds))

        thetaFinal = np.zeros((nruns, nvars))
        LossFinal = np.zeros(nruns)

        # perform optimization over multiple random initial conditions
        for k in tqdm(range(nruns), position=0, leave=True, desc='Finding MLE...', disable=not printing):
            if 'guess' in kwargs:
                guess = np.array(input_guess)
            else:
                if 'random_range' in kwargs:
                    guess = np.random.uniform(0, 10 ** random_range, nvars)
                else:
                    guess = np.random.uniform(0, 100, nvars)

            if 'randomizer' in kwargs and 'guess' in kwargs:  # Use randomizer on input guess
                if randomizer == 'bounded' and 'bounds' in kwargs:
                    guess = np.random.uniform(0.01, 0.99, nvars) * (ub - lb) + lb
                else:
                    if 'random_range' in kwargs:
                        random_guess = np.random.uniform(0, 10 ** random_range, nvars)  # Specify decade for range of random array
                    else:
                        random_guess = np.random.uniform(0, 1, nvars)
                    new_guess = random_guess * randomizer + guess * (1 - randomizer)
                    guess = new_guess

            if 'bounds' in kwargs:
                Result = minimize(obj_function, guess, args=self.args, method='Powell', options=opts, bounds=bds)
            else:
                Result = minimize(obj_function, guess, args=self.args, method='Powell', options=opts)

            thetaFinal[k, :] = Result.x
            LossFinal[k] = Result.fun
            idx = np.argmin(LossFinal)
            if self.fit_function is not None:
                mle_fit = function_clean(thetaFinal[idx, :], self.x, self.fit_function)
            else:
                mle_fit = None
            theta_mle = thetaFinal[idx, :]
            self.idx = idx
            self.thetaFinal = thetaFinal
            self.theta = theta_mle
            self.fit = mle_fit
            self.LossFinalP = LossFinal
            self.ElapsedTime = time.time() - t

        return self


class MLE_old2(object):
    """
    Modified so that all function arguments (including x array, y array, fit function) are included in 'args' list for full customization with
    different fitting situations and customizable objective functions
    """
    def __init__(self, nvars, obj_function, nruns=1, var_names=None, args=None, clipboard_result=True, **kw):
        self.nvars = nvars
        self.obj_function = obj_function
        self.nruns = nruns
        self.args = args
        self.kwargs = kw
        self.idx = []
        self.thetaFinal = []
        self.theta = []
        self.LossFinal = []
        self.ElapsedTime = []
        self.var_names = var_names
        self.clipboard_result = clipboard_result
        self.find_MLE(nruns=self.nruns, nvars=self.nvars, obj_function=self.obj_function, **self.kwargs)

    def find_MLE(self, nruns, nvars, obj_function, printing=True, display=True, **kwargs):
        np.seterr(divide='ignore', invalid='ignore', over='ignore')
        t = time.time()
        opts = {'maxiter': 10000,
                'disp': display,
                #         'maxfun'  : 10000,
                #         'disp' : True,
                # 'full_output': True,
                # 'ftol': 1e-15,
                #         'ftol' : 1e-14,
                # 'eps': 1e-15
                }  # default value.

        if 'guess' in kwargs:
            input_guess = kwargs['guess']

        if 'randomizer' in kwargs:
            randomizer = kwargs['randomizer']

        if 'random_range' in kwargs:
            random_range = kwargs['random_range']

        if 'bounds' in kwargs:
            bounds = kwargs['bounds']
            lb = np.array(bounds[0])
            ub = np.array(bounds[1])
            bds = np.array((lb, ub)).T
            bds = tuple(map(tuple, bds))

        thetaFinal = np.zeros((nruns, nvars))
        LossFinal = np.zeros(nruns)

        # perform optimization over multiple random initial conditions
        for k in tqdm(range(nruns), position=0, leave=True, desc='Finding MLE...', disable=not printing):
            if 'guess' in kwargs:
                guess = np.array(input_guess)
            else:
                if 'random_range' in kwargs:
                    guess = np.random.uniform(0, 10 ** random_range, nvars)
                else:
                    guess = np.random.uniform(0, 100, nvars)

            if 'randomizer' in kwargs and 'guess' in kwargs:  # Use randomizer on input guess
                if randomizer == 'bounded' and 'bounds' in kwargs:
                    guess = np.random.uniform(0.01, 0.99, nvars) * (ub - lb) + lb
                else:
                    if 'random_range' in kwargs:
                        random_guess = np.random.uniform(0, 10 ** random_range, nvars)  # Specify decade for range of random array
                    else:
                        random_guess = np.random.uniform(0, 1, nvars)
                    new_guess = random_guess * randomizer + guess * (1 - randomizer)
                    guess = new_guess

            if 'bounds' in kwargs:
                Result = minimize(obj_function, guess, args=self.args, method='Powell', options=opts, bounds=bds)
            else:
                Result = minimize(obj_function, guess, args=self.args, method='Powell', options=opts)

            thetaFinal[k, :] = Result.x
            LossFinal[k] = Result.fun

        idx = np.argmin(LossFinal)
        theta_mle = thetaFinal[idx, :]
        self.idx = idx
        self.thetaFinal = thetaFinal
        self.theta = theta_mle
        self.LossFinal = LossFinal
        self.ElapsedTime = time.time() - t

        percent_diff_theta(theta_mle, )

        # with np.printoptions(precision=5, suppress=True):
        #     print(self.theta)
        if self.var_names is not None:
            self.df = pd.DataFrame([self.theta], columns=self.var_names)
        else:
            self.df = pd.DataFrame([self.theta])

        if self.clipboard_result:
            self.df.to_clipboard()

class MLE(object):
    """
    Modified so that all function arguments (including x array, y array, fit function) are included in 'args' list for full customization with
    different fitting situations and customizable objective functions
    """
    def __init__(self, theta_in, args, obj_function, nruns=1, var_names=None, clipboard_result=True,
                 guess='init', randomizer='bounded', bounds=True, printing=False, display=True
                 ):
        self.theta_in = theta_in
        self.args = args
        self.obj_function = obj_function
        self.nruns = nruns
        self.var_names = var_names
        self.clipboard_result = clipboard_result
        self.guess = guess
        self.randomizer = randomizer
        self.bounds = bounds
        self.printing = printing
        self.display = display

        self.idx = []
        self.thetaFinal = []
        self.theta = []
        self.LossFinal = []
        self.ElapsedTime = []
        self.find_MLE()

    def find_MLE(self):
        np.seterr(divide='ignore', invalid='ignore', over='ignore')
        t = time.time()
        opts = {'maxiter': 10000,
                'disp': self.display,
                #         'maxfun'  : 10000,
                #         'disp' : True,
                # 'full_output': True,
                # 'ftol': 1e-15,
                #         'ftol' : 1e-14,
                # 'eps': 1e-15
                }  # default value.

        nvars = len(self.theta_in['init'])

        if self.bounds:
            lb = np.array(self.theta_in['lb'])
            ub = np.array(self.theta_in['ub'])
            bds = np.array((lb, ub)).T
            bds = tuple(map(tuple, bds))

        thetaFinal = np.zeros((self.nruns, nvars))
        LossFinal = np.zeros(self.nruns)

        # perform optimization over multiple random initial conditions
        for k in tqdm(range(self.nruns), position=0, leave=True, desc='Finding MLE...', disable=not self.printing):
            if self.guess == 'init':
                guess = np.array(self.theta_in['init'])
            else:
                if self.randomizer == 'bounded':
                    guess = np.random.uniform(0.01, 0.99, nvars) * (ub - lb) + lb
                elif self.randomizer == 'random':
                    random_guess = np.random.uniform(0, 1, nvars)
                    guess = random_guess
                else:
                    guess = np.random.uniform(0, 100, nvars)

            if self.bounds:
                Result = minimize(self.obj_function, guess, args=self.args, method='Powell', options=opts, bounds=bds)
            else:
                Result = minimize(self.obj_function, guess, args=self.args, method='Powell', options=opts)

            thetaFinal[k, :] = Result.x
            LossFinal[k] = Result.fun

        idx = np.argmin(LossFinal)
        theta_mle = thetaFinal[idx, :]
        self.idx = idx
        self.thetaFinal = thetaFinal
        self.theta = theta_mle
        self.LossFinal = LossFinal
        self.ElapsedTime = time.time() - t

        self.theta_diff = percent_diff_theta(theta_mle, self.theta_in, self.var_names)

        if self.var_names is not None:
            self.df = pd.DataFrame([self.theta], columns=self.var_names)
        else:
            self.df = pd.DataFrame([self.theta])

        if self.clipboard_result:
            self.df.to_clipboard()


def function_clean(theta, x, function): # function output (clean)
    w = np.array(theta)
    f = eval(function)
    return f

def lossP(theta, x, y0, function): # mle loss function with Poisson noise assumption
    w = np.array(theta)
    y = eval(function)
    return np.sum(y-(y0)*np.log(y+1e-13))

def lossL(theta, x, y0, function): # mle loss function with Gaussian noise (least squares)
    w = np.array(theta)
    y = eval(function)
    return 0.5*np.sum(((y0)-y)**2)

def lossPL(theta, x, y0, function): # mle loss function with Gaussian + Poisson noise (experimental)
    w = np.array(theta)
    y = eval(function)
    return np.sum(y-(y0)*np.log(y+1e-13)) + 0.5*np.sum(((y0)-y)**2)

def loss_AutoCorr_MSE(theta, upsampled_e, upsampled_ζ, ζ, y0, function):
    """
    :param theta: parameters to generate peaks using energy vector
    :param x: spectral correlation energy differences (meV)
    :param y0: spectral correlation amplitude
    :param function: input function for one or multiple (Gaussian) peaks to be autocorrelated

    - First uses the spectral correlation energy differences to generate an energy array for the spectrum (in such a
      way to ensure the array is the same length as the spectral correlation)
    - Then uses evect, theta, and function to generate a spectrum of peaks
    - Spectrum is autocorrelated and the mean-squared error with respect to the spectral correlation is calculated and
      used in the minimization routine

    """
    spectrum = make_spectrum(upsampled_e, theta, function)
    ycorr = correlate(spectrum, spectrum)
    ycorr = ycorr/max(ycorr)
    ycorr = ycorr + theta[-1]
    ycorr = ycorr/max(ycorr)
    y     = np.interp(upsampled_ζ, ζ, y0)
    y     = y/max(y)

    return 0.5*np.sum((y-ycorr)**2)

def loss_AutoCorrFFT_MSE(theta, x, y0, δ, function):
    """
    :param theta: parameters to generate peaks using energy vector
    :param x: Interferogram stage dealy or optical delay
    :param y0: Interferogram amplitude
    :param function: input function for one or multiple (Gaussian) peaks to be autocorrelated and FFT'd

    - First uses the spectral correlation energy differences to generate an energy array for the spectrum (in such a
      way to ensure the array is the same length as the spectral correlation)
    - Then uses evect, theta, and function to generate a spectrum of peaks
    - Spectrum is autocorrelated and the mean-squared error with respect to the spectral correlation is calculated
    - Autocorrelation

    """
    # upsampled_x = np.linspace(x[0], x[-1], len(x)*10)
    # spectrum = make_spectrum(upsampled_x, theta, function)
    # autocorr = correlate(spectrum, spectrum)
    # int = spec_corr_to_interferogram_FFT(autocorr)
    # upsampled_range = np.linspace(0, 1, 2*len(upsampled_x)-1)
    # downsampled_range = np.linspace(0, 1, len(x))
    # int_downsample = np.interp(downsampled_range, upsampled_range, int)

    ycorr = autocorrelate_peaks(x, theta, function) # First perform autocorrelation of peaks
    int_y = spec_corr_to_interferogram_FFT(ycorr)
    # int_y = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ycorr)))) # Then FFT back into an interferogram

    # eV2cm = 8065.54429
    # cm2eV = 1 / eV2cm
    #
    # # delta = (max(path_length_difference) - min(path_length_difference)) / (N - 1)
    # # zeta_eV = np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000  # in meV
    #
    # N   = len(zeta)
    # Delta = (max(zeta) - min(zeta)) / N
    # increment = 1/Delta
    # path_difference = np.linspace(-0.5*increment, 0.5*increment, N) * cm2eV * 1000 # converted to meV


    # int_y = int_y[ceil(len(int_y) / 2):] # Second half of interferogram

    return 0.5*np.sum((y0-int_y)**2)

def loss_AutoCorrFFT(theta, upsampled_e, upsampled_δ, δ, interferogram, function):
    spectrum = make_spectrum(upsampled_e, theta, function)
    ycorr = correlate(spectrum, spectrum)
    ycorr = ycorr/max(ycorr)
    int_y = spec_corr_to_interferogram_FFT(ycorr)
    y = np.interp(δ, upsampled_δ, int_y)
    y = y/max(y)
    return 0.5*np.sum((y-interferogram)**2)

def gauss_peak(x, x0, gauss_fwhm):
    return (gauss_fwhm/(2*np.sqrt(2*np.log(2))))*np.exp(-(4*np.log(2)/(gauss_fwhm**2))*(x-x0)**2)

def loss_interferogram_wiener(theta, upsampled_e, upsampled_δ, δ, interferogram, function):
    # Make spectrum and its autocorrelation
    spectrum = make_spectrum(upsampled_e, theta, function)
    ycorr = correlate(spectrum, spectrum)
    ycorr = ycorr/max(ycorr)

    # Make Gaussian peak for Wiener diffusion
    w = upsampled_e
    n = (len(w)-1)*2
    delta = w[1] - w[0] # energy difference
    ζ = np.linspace(-n/2*delta, n/2*delta, ceil(n)+1)
    gauss = gauss_peak(ζ, 0, theta[-1])
    gauss = gauss/max(gauss)

    # Convolve
    spec_corr = np.convolve(ycorr, gauss, mode='same')
    spec_corr = spec_corr/max(spec_corr)

    int_y = spec_corr_to_interferogram_FFT(spec_corr)
    y = np.interp(δ, upsampled_δ, int_y)
    y = y/max(y)
    return 0.5*np.sum((y-interferogram)**2)

def make_spectrum(x, theta, function):
    w = np.array(theta)
    f = eval(function)
    return f


def autocorrelate_peaks(x, theta, function):
    n = (len(x) - 1) / 2
    delta = x[1] - x[0] # energy difference
    x = np.linspace(-n/2*delta, n/2*delta, ceil(n)+1) # energy array, replaces x since eval(function) looks for x variable
    w = np.array(theta)
    y = eval(function)
    ycorr = correlate(y, y) + w[-1]
    return ycorr

def spec_corr_to_interferogram_FFT(spec_corr):
    return np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spec_corr))))
    # return np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spec_corr))))/len(spec_corr)
    # return np.fft.ifft(np.fft.ifftshift(spec_corr))

def interferogram_to_spec_corr_FFT(interferogram):
    return np.abs(np.fft.fftshift(np.fft.fft(interferogram)))
    # return np.fft.fftshift(np.fft.fft(interferogram))

def percent_diff_theta(theta_fit, theta_dict, var_names, precision=1):
    lb = theta_dict['lb']
    ub = theta_dict['ub']
    diff_lower = ((np.abs(theta_fit - lb) / ((theta_fit + lb)/2))*100).round(precision)
    diff_upper = ((np.abs(theta_fit - ub) / ((theta_fit + ub)/2))*100).round(precision)
    diff_frame = pd.DataFrame([diff_lower, diff_upper], columns=var_names)
    diff_frame.index = ['lb', 'ub']
    return diff_frame

# y_int_FFT = np.fft.fftshift(np.fft.fft(y_int))
# y_int_FFT_and_back = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(y_int))))
# plt.plot(np.fft.ifft(np.fft.ifftshift(y_int_FFT)))
