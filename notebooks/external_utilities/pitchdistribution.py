# -*- coding: utf-8 -*-
from __future__ import division
import essentia
import essentia.standard as std
import numpy as np
import json
import copy
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt
from external_utilities.converter import Converter
import numbers
import pickle
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PitchDistribution(object):
    def __init__(self, pd_bins, pd_vals, kernel_width=7.5, ref_freq=440.0):
        """-------------------------------------------------------------------
        The main data structure that wraps all the relevant information about a
        pitch distribution.
        ----------------------------------------------------------------------
        pd_bins      : Bins of the pitch distribution. It is a 1-D list of
                       equally spaced monotonically increasing frequency
                       values.
        pd_vals      : Values of the pitch distribution
        kernel_width : The standard deviation of the Gaussian kernel. See
                       generate_pd() of ModeFunctions for more detail.
        ref_freq     : Reference frequency that is used while generating the
                       distribution. If the tonic of a recording is annotated,
                       this is variable that stores it.
        --------------------------------------------------------------------"""
        assert len(pd_bins) == len(pd_vals), 'Lengths of bins and vals are ' \
                                             'different.'

        self.bins = np.array(pd_bins)  # force numpy array
        self.vals = np.array(pd_vals)  # force numpy array
        self.kernel_width = kernel_width
        if ref_freq is None:
            self.ref_freq = None
        else:
            self.ref_freq = np.array(ref_freq)  # force numpy array

    @property
    def step_size(self):
        # get step size in cents
        if self.has_hz_bin():
            temp_ss = Converter.hz_to_cent(self.bins[1], self.bins[0])
        else:  # has_cent_bin
            temp_ss = self.bins[1] - self.bins[0]

        # TEMPORARY FIX: round step_size to one decimal point
        return temp_ss if temp_ss == (round(temp_ss * 10) / 10) \
            else round(temp_ss * 10) / 10

    @property
    def bin_unit(self):
        err_str = 'Invalid reference. ref_freq should be either None ' \
                  '(bin unit is Hz) or a number greater than 0.'
        if self.ref_freq is None:
            return 'Hz'
        elif isinstance(self.ref_freq, (numbers.Number, np.ndarray)):
            assert self.ref_freq > 0, err_str
            return 'cent'
        else:
            return ValueError(err_str)

    @staticmethod
    def from_cent_pitch(cent_track, ref_freq=440.0, kernel_width=7.5,
                        step_size=7.5, norm_type='sum'):
        """--------------------------------------------------------------------
        Given the pitch track in the unit of cents, generates the Pitch
        Distribution of it. the pitch track from a text file. 0th column is the
        time-stamps and
        1st column is the corresponding frequency values.
        -----------------------------------------------------------------------
        cent_track:     1-D array of frequency values in cents.
        ref_freq:       Reference frequency used while converting Hz values to
                        cents.
                        This number isn't used in the computations, but is to
                        be recorded in the PitchDistribution object.
        kernel_width:  The standard deviation of the gaussian kernel, used in
                        Kernel Density Estimation. If 0, a histogram is given
        step_size:      The step size of the Pitch Distribution bins.
        --------------------------------------------------------------------"""
        assert step_size > 0, 'The step size should have a positive value'

        # Some extra interval is added to the beginning and end since the
        # superposed Gaussian for kernel_width would introduce some tails in
        # the ends. These vanish after 3 sigmas(=kernel_width).

        # The limits are also quantized to be a multiple of chosen step-size
        # kernel_width = standard deviation of the gaussian kernel
        # parse the cent_track
        try:
            cent_track = np.loadtxt(cent_track)
        except ValueError:
            logger.debug('cent_track is already a numpy array')

        if cent_track.ndim > 1:  # pitch is given as [time, pitch, (conf)]
            cent_track = cent_track[:, 1]

        # filter out NaN, and infinity
        cent_track = cent_track[~np.isnan(cent_track)]
        cent_track = cent_track[~np.isinf(cent_track)]

        # Finds the endpoints of the histogram edges. Histogram bins will be
        # generated as the midpoints of these edges.
        min_edge = min(cent_track) - (step_size / 2.0)
        max_edge = max(cent_track) + (step_size / 2.0)
        pd_edges = np.concatenate(
            [np.arange(-step_size / 2.0, min_edge, -step_size)[::-1],
             np.arange(step_size / 2.0, max_edge, step_size)])

        # An exceptional case is when min_bin and max_bin are both positive
        # In this case, pd_edges would be in the range of [step_size/2, max_
        # bin]. If so, a -step_size is inserted to the head, to make sure 0
        # would be in pd_bins. The same procedure is repeated for the case
        # when both are negative. Then, step_size is inserted to the tail.
        pd_edges = pd_edges if -step_size / 2.0 in pd_edges else np.insert(
            pd_edges, 0, -step_size / 2.0)
        pd_edges = pd_edges if step_size / 2.0 in pd_edges else np.append(
            pd_edges, step_size / 2.0)

        # Generates the histogram and bins (i.e. the midpoints of edges)
        pd_vals, pd_edges = np.histogram(cent_track, bins=pd_edges,
                                         density=False)
        pd_bins = np.convolve(pd_edges, [0.5, 0.5])[1:-1]  # the bin centers

        # initialize the distribution
        pd = PitchDistribution(pd_bins, pd_vals, kernel_width=0,
                               ref_freq=ref_freq)
        pd.smoothen(kernel_width=kernel_width)

        # normalize
        pd.normalize(norm_type=norm_type)

        return pd

    def smoothen(self, kernel_width=7.5):
        if kernel_width > 0:
            # smooth the histogram
            normal_dist = scipy.stats.norm(loc=0, scale=kernel_width)
            xn = np.concatenate(
                [np.arange(0, - 5 * kernel_width, -self.step_size)[::-1],
                 np.arange(self.step_size, 5 * kernel_width, self.step_size)])
            sampled_norm = normal_dist.pdf(xn)
            if len(sampled_norm) <= 1:
                raise ValueError(
                    "the smoothing factor is too small compared to the step "
                    "size, such that the convolution kernel returns a single "
                    "point Gaussian. Either increase the value to at least "
                    "(step size/3) or assign kernel width to 0, for no "
                    "smoothing.")
            # convolution generates tails
            extra_num_bins = np.floor(len(sampled_norm) / 2)

            self.bins = np.concatenate(
                (np.arange(self.bins[0] - extra_num_bins * self.step_size,
                           self.bins[0], self.step_size), self.bins,
                 np.arange(self.bins[-1] + self.step_size, self.bins[-1] +
                           extra_num_bins * self.step_size + self.step_size,
                           self.step_size)))
            self.vals = np.convolve(self.vals, sampled_norm)
            assert len(self.bins) == len(self.vals), 'Lengths of bins and ' \
                                                     'vals are different.'
            self.kernel_width = (kernel_width if self.kernel_width == 0 else
                                 self.kernel_width * kernel_width)

    @staticmethod
    def from_hz_pitch(hz_track, ref_freq=440.0, kernel_width=7.5,
                      step_size=7.5, norm_type='sum'):
        try:
            hz_track = np.loadtxt(hz_track)
        except ValueError:
            logger.debug('hz_track is already a numpy array')

        if hz_track.ndim > 1:  # pitch is given as [time, pitch, (conf)] array
            hz_track = hz_track[:, 1]

        # filter out the NaN, -infinity and +infinity and values < 20
        hz_track = hz_track[~np.isnan(hz_track)]
        hz_track = hz_track[~np.isinf(hz_track)]
        hz_track = hz_track[hz_track >= 20.0]
        cent_track = Converter.hz_to_cent(hz_track, ref_freq, min_freq=20.0)

        return PitchDistribution.from_cent_pitch(
            cent_track, ref_freq=ref_freq, kernel_width=kernel_width,
            step_size=step_size, norm_type=norm_type)

    def __eq__(self, other):
        eq_bool = True
        self_dict = self.__dict__
        other_dict = other.__dict__

        # numpy array need to be compared with np.allclose
        eq_bool = eq_bool and np.allclose(self_dict.pop("vals", None),
                                          other_dict.pop("vals", None))
        eq_bool = eq_bool and np.allclose(self_dict.pop("bins", None),
                                          other_dict.pop("bins", None))

        return eq_bool and self_dict == other_dict

    def is_pcd(self):
        """--------------------------------------------------------------------
        The boolean flag of whether the instance is PCD or not.
        --------------------------------------------------------------------"""
        if self.has_cent_bin():  # cent bins; compare directly
            return np.isclose(max(self.bins) - min(self.bins),
                              1200 - self.step_size)
        else:
            dummy_d = copy.deepcopy(self)

            dummy_d.hz_to_cent(dummy_d.bins[0])

            return np.isclose(max(dummy_d.bins) - min(dummy_d.bins),
                              1200 - dummy_d.step_size)

    def is_pdf(self):
        return np.isclose(np.sum(self.vals), 1)

    def distrib_type(self):
        return 'pcd' if self.is_pcd() else 'pd'

    def has_hz_bin(self):
        return self.bin_unit in ['hz', 'Hz', 'Hertz', 'hertz']

    def has_cent_bin(self):
        return self.bin_unit in ['cent', 'Cent', 'cents', 'Cents']

    def normalize(self, norm_type='sum'):
        if norm_type is None:  # nothing, keep the occurrences (histogram)
            normval = 1
        elif norm_type == 'area':  # area under the curve using simpsons rule
            normval = scipy.integrate.simps(self.vals, dx=self.step_size)
        elif norm_type == 'sum':  # sum normalization
            normval = np.sum(self.vals)
        elif norm_type == 'max':  # max number becomes 1
            normval = max(self.vals)
        else:
            raise ValueError("norm_type can be None, 'area', 'sum' or 'max'")

        self.vals = self.vals / normval

    def detect_peaks(self, min_peak_ratio=0.15):
        """--------------------------------------------------------------------
        Finds the peak indices of the distribution. These are treated as tonic
        candidates in higher order functions.
        min_peak_ratio: The minimum ratio between the max peak value and the
                        value of a detected peak
        --------------------------------------------------------------------"""
        assert 1 >= min_peak_ratio >= 0, \
            'min_peak_ratio should be between 0 (keep all peaks) and ' \
            '1 (keep only the highest peak)'

        # Peak detection is handled by Essentia
        detector = std.PeakDetection()
        peak_bins, peak_vals = detector(essentia.array(self.vals))

        # Essentia normalizes the positions to 1, they are converted here
        # to actual index values to be used in bins.
        peak_inds = np.array([int(round(bn * (len(self.bins) - 1)))
                              for bn in peak_bins])

        # if the object is pcd and there is a peak at zeroth index,
        # there will be another in the last index. Since a pcd is circular
        # remove the lower value
        if self.is_pcd() and peak_inds[0] == 0:
            if peak_vals[0] >= peak_vals[-1]:
                peak_inds = peak_inds[:-1]
                peak_vals = peak_vals[:-1]
            else:
                peak_inds = peak_inds[1:]
                peak_vals = peak_vals[1:]

        # remove peaks lower than the min_peak_ratio
        peak_bool = peak_vals / max(peak_vals) >= min_peak_ratio

        return peak_inds[peak_bool], peak_vals[peak_bool]

    def to_pcd(self):
        """--------------------------------------------------------------------
        Given the pitch distribution of a recording, generates its pitch class
        distribution, by octave wrapping.
        -----------------------------------------------------------------------
        pD: PitchDistribution object. Its attributes include everything we need
        --------------------------------------------------------------------"""
        assert not self.is_pcd(), 'The object is already a PCD'

        has_hz_bin = self.has_hz_bin()  # remember the bin unit for later
        if self.has_hz_bin():
            self.hz_to_cent(self.bins[0])

        # Initializations
        pcd_bins = np.arange(0, 1200, self.step_size)
        pcd_vals = np.zeros(len(pcd_bins))

        # Octave wrapping
        for bb, vv in zip(self.bins, self.vals):

            idx = int(round((bb % 1200) / self.step_size))
            idx = idx if idx != 160 else 0
            pcd_vals[idx] += vv

        self.bins = pcd_bins
        self.vals = pcd_vals

        assert len(pcd_bins) == len(pcd_vals), 'Lengths of bins and vals ' \
                                               'are different.'

        # convert the unit back to what is was
        if has_hz_bin:
            self.cent_to_hz()

    def hz_to_cent(self, ref_freq):
        if self.has_hz_bin():
            self.bins = Converter.hz_to_cent(self.bins, ref_freq)
            self.ref_freq = ref_freq

            # make sure all the bins stay between 0 - 1200 for PCDs
            if self.is_pcd():
                self.bins = np.mod(self.bins, 1200)

                idx = np.argsort(self.bins)
                self.bins = self.bins[idx]
                self.vals = self.vals[idx]
        else:
            raise ValueError('The bin unit should be "hz".')

    def cent_to_hz(self):
        if self.has_cent_bin():
            self.bins = Converter.cent_to_hz(self.bins, self.ref_freq)
            self.ref_freq = None
        else:
            raise ValueError('The bin unit should be "cent".')

    def shift(self, shift_idx):
        """--------------------------------------------------------------------
        Shifts the distribution by the given number of samples
        -----------------------------------------------------------------------
        shift_idx : The number of samples that the distribution is to be
                    shifted
        --------------------------------------------------------------------"""
        # Shift only if the index is non-zero and the distribution is in
        # cent units
        if shift_idx and self.has_cent_bin():
            # update reference frequency
            self.ref_freq = Converter.cent_to_hz(
                self.bins[shift_idx] - self.bins[0],
                ref_freq=self.ref_freq)

            # If distribution is a PCD, we do a circular shift
            if self.is_pcd():
                self.vals = np.concatenate((self.vals[shift_idx:],
                                            self.vals[:shift_idx]))
            else:  # If distribution is a PD, shift the bins.
                self.bins -= self.step_size * shift_idx

    def merge(self, distrib):
        """
        Merges the distribution with another distribution
        :param distrib: input distribution (PD or PCD)
        """
        assert self.bin_unit == distrib.bin_unit, \
            'The bin units of the compared distributions should match.'
        assert self.distrib_type() == distrib.distrib_type(), \
            'The features should be of the same type'
        assert self.step_size == distrib.step_size, \
            'The step_sizes should be the same'
        assert self.is_pdf() == distrib.is_pdf(), \
            'The normalization should be the same'

        # find the max and min bins
        min_bin = np.min([np.min(self.bins), np.min(distrib.bins[0])])
        max_bin = np.max([np.max(self.bins[-1]), np.max(distrib.bins[-1])])

        # initialize the bins and vals
        bins = np.arange(min_bin, max_bin + self.step_size / 2.0,
                         self.step_size)
        assert 0 in bins, 'Zero should be in the bins'
        vals = np.zeros(len(bins))

        # add the vals in the distributions to the corresponding bins
        for dd in (self, distrib):
            bin_bool = np.logical_and(bins >= np.min(dd.bins),
                                      bins <= np.max(dd.bins))
            vals[bin_bool] += dd.vals

        # update self
        is_pdf = self.is_pdf()  # record if pdf
        self.bins = bins
        self.vals = vals
        if is_pdf:
            self.normalize()

    def plot(self):
        plt.plot(self.bins, self.vals)
        self.label_figure()

    def bar(self):
        bars = plt.bar(self.bins, self.vals, width=self.step_size,
                       align='center')
        self.label_figure()

        return bars

    def label_figure(self):
        if self.is_pcd():
            plt.title('Pitch class distribution')
            ref_freq_str = 'Hz x 2^n'
        else:
            plt.title('Pitch distribution')
            ref_freq_str = 'Hz'
        if self.has_hz_bin():
            plt.xlabel('Frequency (Hz)')
        else:
            plt.xlabel('Normalized Frequency (cents), ref = {0}{1}'.format(
                str(self.ref_freq), ref_freq_str))
        plt.ylabel('Occurence')

    @staticmethod
    def from_pickle(input_str):
        try:  # file given
            return pickle.load(open(input_str, 'rb'))
        except IOError:  # string given
            return pickle.loads(input_str, 'rb')

    def to_pickle(self, file_name=None):
        if file_name is None:
            return pickle.dumps(self)
        else:
            pickle.dump(self, open(file_name, 'wb'))

    @staticmethod
    def from_json(file_name):
        """--------------------------------------------------------------------
        Loads a PitchDistribution object from JSON file.
        -----------------------------------------------------------------------
        file_name    : The filename of the JSON file
        --------------------------------------------------------------------
        """
        try:
            distrib = json.load(open(file_name, 'r'))
        except IOError:  # json string
            distrib = json.loads(file_name)

        distrib = distrib if isinstance(distrib, dict) else distrib[0]

        return PitchDistribution.from_dict(distrib)

    def to_json(self, file_name=None):
        """--------------------------------------------------------------------
        Saves the PitchDistribution object to a JSON file.
        -----------------------------------------------------------------------
        file_name    : The file path of the JSON file to be created.
        --------------------------------------------------------------------"""
        dist_json = self.to_dict()

        if file_name is None:
            return json.dumps(dist_json, indent=4)
        else:
            json.dump(dist_json, open(file_name, 'w'), indent=4)

    @staticmethod
    def from_dict(distrib_dict):
        return PitchDistribution(distrib_dict['bins'], distrib_dict['vals'],
                                 kernel_width=distrib_dict['kernel_width'],
                                 ref_freq=distrib_dict['ref_freq'])

    def to_dict(self):
        pdict = self.__dict__
        for key in pdict.keys():
            try:
                # convert to list from np array
                pdict[key] = pdict[key].tolist()
            except AttributeError:
                pass

        return pdict
