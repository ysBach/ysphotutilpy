"""
Currently only the half-wave plate angle (HWP angle) of 0, 22.5, 45,
67.5 combination is available.

Primitive naming:
<lin/circ/all>_<oe/sr>_<n>set
- lin/circ/all: linear, circular, or all Stoke's parameter will be
  determined.
- oe/sr: o- and e-ray (e.g., Wollaston or Savart prism is used) or
  single-ray version
- n: the number of HWP angles used. 3 or 4.
"""
import numpy as np

__all__ = ['LinPolOE4', 'proper_pol']


def sqsum(v1, v2):
    return v1**2 + v2**2


def eprop(*errs):
    var = 0
    for e in errs:
        var += e**2
    return np.sqrt(var)


class PolObjMixin:
    def _set_qu(self):
        if self.mode == "lin_oe_4set":
            # The ratios, r for each HWP angle
            self.r000 = self.i000_e/self.i000_o
            self.r045 = self.i450_e/self.i450_o
            self.r225 = self.i225_e/self.i225_o
            self.r675 = self.i675_e/self.i675_o

            # noise-to-signal (dr/r) of I_e/I_o for each HWP angle
            self.ns000 = eprop(self.di000_e/self.i000_e,
                               self.di000_o/self.i000_o)
            self.ns450 = eprop(self.di450_e/self.i450_e,
                               self.di450_o/self.i450_o)
            self.ns225 = eprop(self.di225_e/self.i225_e,
                               self.di225_o/self.i225_o)
            self.ns675 = eprop(self.di675_e/self.i675_e,
                               self.di675_o/self.i675_o)

            # The q/u values
            self.r_q = np.sqrt(self.r000/self.r045)
            self.r_u = np.sqrt(self.r225/self.r675)
            self.q0 = (self.r_q - 1)/(self.r_q + 1)
            self.u0 = (self.r_u - 1)/(self.r_u + 1)

            # The errors
            s_q = eprop(self.ns000, self.ns450)
            s_u = eprop(self.ns225, self.ns675)
            self.dq0 = (self.r_q/(self.r_q + 1)**2 * s_q)
            self.du0 = (self.r_u/(self.r_u + 1)**2 * s_u)
        else:
            raise ValueError(f"{self.mode} not understood.")


'''
    @property
    def _set_qu(self):
        self.o_val = {}
        self.e_val = {}
        self.do_val = {}
        self.de_val = {}
        self.ratios = {}
        self.v_ratios = {}  # variances, not standard deviations
        self.q = {}
        self.u = {}
        self.dq = {}
        self.du = {}
        self.step = 0
        self.messages = {}

        for hwp in HWPS:
            idx = np.where(self.order == hwp)
            o_vals = self.orays[:, idx]
            e_vals = self.erays[:, idx]
            do_vals = self.dorays[:, idx]
            de_vals = self.derays[:, idx]
            self.o_val[hwp] = o_vals
            self.e_val[hwp] = e_vals
            self.do_val[hwp] = do_vals
            self.de_val[hwp] = de_vals
            self.ratios[hwp] = e_vals / o_vals
            self.v_ratios[hwp] = (do_vals/o_vals)**2 + (de_vals/e_vals)**2

        self.r_q = np.sqrt(self.ratios[HWPS[0]]/self.ratios[HWPS[2]])
        self.r_u = np.sqrt(self.ratios[HWPS[1]]/self.ratios[HWPS[3]])
        self.q[0] = (self.r_q - 1)/(self.r_q + 1)
        self.u[0] = (self.r_u - 1)/(self.r_u + 1)
        self.dq[0] = (self.r_q / (self.r_q + 1)**2
                      * np.sqrt(self.v_ratios[HWPS[0]]
                                + self.v_ratios[HWPS[2]])
                      )
        self.du[0] = (self.r_u / (self.r_u + 1)**2
                      * np.sqrt(self.v_ratios[HWPS[1]]
                                + self.v_ratios[HWPS[3]])
                      )
        self.messages[0] = (f"Initialize with input data (order: {self.order})"
                            + f" for {self.ndata} data-sets.")

    def set_check_2d_4(self, name, arr, ifnone=None):
        if (ifnone is not None) and (arr is None):
            a = ifnone
        else:
            a = np.atleast_2d(arr)
            if a.shape[1] != 4:
                raise ValueError(f"{name} must be a length 4 or (N, 4) array.")

        setattr(self, name, a)
'''


class LinPolOE4(PolObjMixin):
    def __init__(self, i000_o, i000_e, i450_o, i450_e,
                 i225_o, i225_e, i675_o, i675_e,
                 di000_o=None, di000_e=None, di450_o=None, di450_e=None,
                 di225_o=None, di225_e=None, di675_o=None, di675_e=None
                 ):
        """
        Parameters
        ----------
        ixxx_[oe] : array-like
            The intensity (in linear scale, e.g., sky-subtracted ADU) in
            the half-wave plate angle of ``xxx/10`` degree in the ``o``
            or ``e``-ray.
        dixxx_[oe] : array-like, optinal
            The 1-sigma error-bars of the corresponding ``ixxx_[oe]``.
            It must have the identical length as ``ixxx_[oe]`` if not
            None.
        """
        self.mode = "lin_oe_4set"
        self.i000_o = np.array(i000_o)
        self.i000_e = np.array(i000_e)
        self.i450_o = np.array(i450_o)
        self.i450_e = np.array(i450_e)
        self.i225_o = np.array(i225_o)
        self.i225_e = np.array(i225_e)
        self.i675_o = np.array(i675_o)
        self.i675_e = np.array(i675_e)

        if not (self.i000_o.shape == self.i000_e.shape
                == self.i450_o.shape == self.i450_e.shape
                == self.i225_o.shape == self.i225_e.shape
                == self.i675_o.shape == self.i675_e.shape):
            raise ValueError("all ixxx_<oe> must share the identical shape.")

        _dis = dict(di000_o=di000_o, di000_e=di000_e,
                    di450_o=di450_o, di450_e=di450_e,
                    di225_o=di225_o, di225_e=di225_e,
                    di675_o=di675_o, di675_e=di675_e)
        for k, v in _dis.items():
            if v is None:
                v = np.zeros_like(getattr(self, k[1:]))
            setattr(self, k, np.array(v))

        if not (self.di000_o.shape == self.di000_e.shape
                == self.di450_o.shape == self.di450_e.shape
                == self.di225_o.shape == self.di225_e.shape
                == self.di675_o.shape == self.di675_e.shape):
            raise ValueError("all dixxx_<oe> must share the identical shape.")

    # TODO: This should apply for any linear polarimetry using HWP..?
    # So maybe should move to Mixin class.
    def calc_pol(self, p_eff=1., dp_eff=0.,
                 q_inst=0., u_inst=0., dq_inst=0., du_inst=0.,
                 rot_instq=0., rot_instu=0.,
                 pa_inst=0., theta_inst=0., dtheta_inst=0.,
                 percent=True, degree=True):
        '''
        Parameters
        ----------
        p_eff, dp_eff : float, optional.
            The polarization efficiency and its error. Defaults to ``1``
            and ``0``.
        q_inst, u_inst, dq_inst, du_inst : float, optional
            The instrumental q (Stokes Q/I) and u (Stokes U/I) values
            and their errors. All defaults to ``0``.
        rot_instq, rot_instu: float, array-like, optional.
            The instrumental rotation. In Nayoro Pirka MSI manual, the
            average of ``INS-ROT`` for the HWP angle 0 and 45 at the
            start and end of exposure (total 4 values) are to be used
            for ``rot_instq``, etc. If array, it must have the same
            length as ``ixxx_[oe]``.
        pa_inst : float, array-like, optional.
            The position angle (North to East) of the instrument.
            If array-like, it must have the same length as
            ``ixxx_[oe]``.
        theta_inst, dtheta_inst : float, optinoal.
            The instrumental polarization rotation angle theta and its
            error.
        percent : bool, optional.
            Whether ``p_eff``, ``dp_eff``, ``q_inst``, ``dq_inst``,
            ``u_inst``, ``du_inst`` are in percent unit. Defaults to
            ``True``.
        degree : bool, optional.
            Whether ``rot_instq``, ``rot_instu``, ``theta_inst``,
            ``dtheta_inst`` are in degree unit. Otherwise it must be in
            radians. Defaults to ``True``.
        '''

        if percent:
            # polarization efficiency
            p_eff = p_eff/100
            dp_eff = dp_eff/100
            # instrumental polarization
            q_inst = q_inst/100
            u_inst = u_inst/100
            dq_inst = dq_inst/100
            du_inst = du_inst/100

        self.p_eff = p_eff
        self.dp_eff = dp_eff
        self.q_inst = q_inst
        self.u_inst = u_inst
        self.dq_inst = dq_inst
        self.du_inst = du_inst

        if degree:
            # instrument's rotation angle from FITS header
            rot_instq = np.deg2rad(rot_instq)
            rot_instu = np.deg2rad(rot_instu)
            # position angle and instrumental polarization angle
            pa_inst = np.deg2rad(pa_inst)
            theta_inst = np.deg2rad(theta_inst)
            dtheta_inst = np.deg2rad(dtheta_inst)

        self.rot_instq = rot_instq
        self.rot_instu = rot_instu
        self.pa_inst = pa_inst
        self.theta_inst = theta_inst
        self.dtheta_inst = dtheta_inst

        self._set_qu()

        self.q1 = self.q0/self.p_eff
        self.u1 = self.u0/self.p_eff
        self.dq1 = eprop(self.dq0, np.abs(self.q1)*self.dp_eff)/self.p_eff
        self.du1 = eprop(self.du0, np.abs(self.u1)*self.dp_eff)/self.p_eff

        # self.messages = ("Polarization efficiency corrected by "
        #                   + f"p_eff = {self.p_eff}, "
        #                   + f"dp_eff = {self.dp_eff}.")

        rotq = (np.cos(2*self.rot_instq), np.sin(2*self.rot_instq))
        rotu = (np.cos(2*self.rot_instu), np.sin(2*self.rot_instu))
        self.q2 = (self.q1 - (self.q_inst*rotq[0] - self.u_inst*rotq[1]))
        self.u2 = (self.u1 - (self.q_inst*rotu[1] + self.u_inst*rotu[0]))
        # dq_inst_rot = eprop(self.dq_inst*rotq[0], self.du_inst*rotq[1])
        # du_inst_rot = eprop(self.dq_inst*rotu[1], self.du_inst*rotu[0])
        self.dq2 = eprop(self.dq1, self.dq_inst*rotq[0], self.du_inst*rotq[1])
        self.du2 = eprop(self.du1, self.dq_inst*rotu[1], self.du_inst*rotu[0])

        theta = self.theta_inst - self.pa_inst
        rot = (np.cos(2*theta), np.sin(2*theta))
        self.q3 = +1*self.q2*rot[0] + self.u2*rot[1]
        self.u3 = -1*self.q2*rot[1] + self.u2*rot[0]
        self.dq3 = eprop(rot[0]*self.dq2,
                         rot[1]*self.du2,
                         2*self.u3*dtheta_inst)
        self.du3 = eprop(rot[1]*self.dq2,
                         rot[0]*self.du2,
                         2*self.q3*dtheta_inst)

        self.pol = np.sqrt(self.q3**2 + self.u3**2)
        self.dpol = eprop(self.q3*self.dq3, self.u3*self.du3)/self.pol
        self.theta = 0.5*np.arctan2(self.u3, self.q3)
        self.dtheta = 0.5*self.dpol/self.pol

        if percent:
            self.pol *= 100
            self.dpol *= 100

        if degree:
            self.theta = np.rad2deg(self.theta)
            self.dtheta = np.rad2deg(self.dtheta)


def proper_pol(pol, theta, psang, degree=True):
    if not degree:
        theta = np.rad2deg(theta)
        psang = np.rad2deg(psang)

    dphi = psang + 90
    dphi = dphi - 180*(dphi//180)
    theta_r = theta - dphi
    pol_r = pol*np.cos(2*np.deg2rad(theta_r))
    if not degree:
        theta_r = np.deg2rad(theta_r)
    return pol_r, theta_r


"""
    @classmethod
    def from1d(self, orays, erays, dorays=None, derays=None,
               order=[0, 22.5, 45, 67.5]):
        '''
        Parameters
        ----------
        orays, erays: 1-d array-like or 2-d of shape ``(N, 4)``
            The oray and eray intensities in ``(N, 4)`` shape. The four
            elements are assumed to be in the order of HWP angle as in
            ``order``.
        order : array-like, optional.
            The HWP angle order in ``orays`` and ``erays``.
        '''
        # orays, erays, dorays, derays to 2-d array
        self.set_check_2d_4('orays', orays)
        self.set_check_2d_4('erays', erays)
        self.set_check_2d_4('dorays', dorays, ifnone=np.zeros_like(self.orays))
        self.set_check_2d_4('derays', derays, ifnone=np.zeros_like(self.erays))

        if not (self.orays.shape == self.erays.shape
                == self.dorays.shape == self.derays.shape):
            raise ValueError("orays, erays, dorays, derays must share "
                             + "the identical shape.")
        self.ndata = self.orays.shape[0]

        # check order
        self.order = np.atleast_1d(order)
        if self.order.ndim > 1:
            raise ValueError("order must be 1-d array-like.")
        elif not all(val in order for val in HWPS):
            raise ValueError(f"order must contain all the four of {HWPS}.")
        elif self.order.shape[0] != 4:
            raise ValueError(f"order must contain only the four of {HWPS}.")

    def correct_peff(self, p_eff, dp_eff=0, percent=True):
        if percent:
            p_eff = p_eff/100
            dp_eff = dp_eff/100
        self.p_eff = p_eff
        self.dp_eff = dp_eff

        self.step += 1
        idx = self.step

        self.q[idx] = self.q/self.p_eff
        self.u[idx] = self.u/self.p_eff
        sig_q = self.dq/self.p_eff
        sig_u = self.du/self.p_eff
        sig_p = self.dp_eff/self.p_eff
        self.dq[idx] = self.q[idx] * np.sqrt(sig_q**2 + sig_p**2)
        self.du[idx] = self.u[idx] * np.sqrt(sig_u**2 + sig_p**2)
        self.messages[idx] = ("Polarization efficiency corrected by "
                              + f"p_eff = {self.p_eff}, "
                              + f"dp_eff = {self.dp_eff}.")


        gain_corrected : bool, optional.
            Whether the values are in ADU (if ``False``) or electrons
            (if ``True``). Defaults to ``False``.
"""
