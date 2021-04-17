import holoviews as hv
from holoviews import opts
import panel as pn
import param
from scipy.special import expit
from scipy import *
from scipy.integrate import ode
from scipy.integrate import odeint
import numpy as np
import sys

q = 1.6e-19  # Electron charge (C)

class LaserRateEquations(param.Parameterized):
  shape = param.ObjectSelector(default='gaussian',
                                objects=['gaussian', 'smooth_step',
                                         'stepdown', 'cw'],
                                doc='Shape')
  pump_current = param.Number(178, doc='Pump current maximum (mA)', step=1)
  pulse_width = param.Number(30, doc='Pump pulse width (ns)', step=1)
  rise_scale = param.Number(30, doc='Rise time scale factor', step=5)
  offset = param.Number(25, doc='Pulse offset (ns)', step=1)
  volume = param.Number(2e-9, doc='Volume (cm^3)', step=1e-9)
  carrier_lifetime = param.Number(2, doc='Carrier relaxation time (τn, ns)')
  gain_slope_constant = param.Number(1.5e-14, doc='Gain slope constant (g0, cm^3/ns)', step=1e-15)
  threshold_density = param.Number(1e18, doc= 'Threshold carrier density (Nth, cm^-3)', step=1e16) 
  compression_factor = param.Number(1e-17, doc='Gain compression factor (ϵ, cm^3)', step=1e-16)
  confinement_factor = param.Number(1, doc='Confinement factor (Γ)', step=0.01)
  photon_lifetime = param.Number(0.001, doc='Cavity photon lifetime (τp, ns)', step=0.0001)
  emission_factor = param.Number(0.0001, doc='Spontaneous emission factor (β)', step=0.0001)
  photon_energy = param.Number(3e-19, doc='Photon energy (J)', step=1e-16)
  simulation_time = param.Number(50, doc='Simulation time (ns)', step=1)
  step_size_resolution = param.Number(0.01, doc='Step size resolution (ns)', step=0.01)

  @param.depends('shape', 'pump_current', 'pulse_width', 'rise_scale', 'offset',
                 'volume', 'carrier_lifetime', 'gain_slope_constant',
                 'threshold_density', 'compression_factor', 'confinement_factor',
                 'photon_lifetime', 'emission_factor', 'photon_energy', 
                 'simulation_time', 'step_size_resolution')
  def view(self):
    params = (self.pump_current,self.pulse_width,self.rise_scale,
              self.offset,self.volume,self.carrier_lifetime,
              self.gain_slope_constant,self.threshold_density,self.compression_factor,
              self.confinement_factor,self.photon_lifetime,self.emission_factor,
              self.photon_energy,self.simulation_time,self.step_size_resolution)
    ts, n, i_p, p = solve_laser(self.shape, params)
    return pn.Column(
      hv.Curve((ts, i_p)).opts(shared_axes=False,
                                    width=1000, height=250,
                                    color='orange',
                                    title='Gain Switching in SDLs',
                                    xlabel= 't (ns)', ylabel='Current [I(t), mA]'),
      hv.Curve((ts, p * 1000)).opts(shared_axes=False, width=1000,
                                    height=250, color='green',
                                    xlabel= 't (ns)', ylabel='Max Power [P(t), mW]'),
      hv.Curve((ts, n)).opts(width=1000, height=250) * hv.Curve(
          (ts, self.threshold_density * np.ones_like(ts))).opts(
              width=1000, height=250,
              xlabel='t (ns)', ylabel='Carrier Conc. [N(t), cm^(-3)]')
    )


# Pump current modes

def laser(pulse_shape, params):
  I, T, scale, T_0, v, τ_n, g_0, N_th, ϵ, Γ, τ_p, β, _, _, _ = params
  i_fn = {
      'gaussian': lambda t: I * np.exp(-((t - T_0) / (T / 3)) ** 2),
      'smooth_step': lambda t: I * (expit((t - T_0 + T / 2) / T * scale) - expit((t - T_0 - T / 2) / T * scale)),
      'stepdown': lambda t: I * (1 - np.heaviside(t - T, 0.5)),
      'cw': lambda t: I
  }[pulse_shape]
  def laser_fn(y, t):
    return [
      (i_fn(t) / (q * v) * 1e-12) - (y[0] / τ_n) -  g_0 * (y[0] - N_th) * (y[1] / (1 + ϵ * y[1])), 
      Γ * g_0 * (y[0] - N_th) * (y[1] / (1 + ϵ * y[1])) - y[1] / τ_p + (Γ * β * y[0])
    ]
  return laser_fn, i_fn

def solve_laser(pulse_shape, params):
  Γ, τ_p, β, hν = params[-6:-2]
  v = params[4]
  laser_fn, i_fn = laser(pulse_shape, params)
  ys, ts = [], []
  simulation_time, dt = params[-2:]
  ts = np.linspace(0, simulation_time, int(simulation_time // dt))
  # optical power I(t)
  current = i_fn(np.asarray(ts))
  # carrier concentration N(t), photon concentration S(t)
  n, s = odeint(laser_fn, (0, 0), ts).T
  # optical power P(t)
  p = s * v * hν / (2 * Γ * τ_p) * 1e9
  return ts, n, current, p


if __name__ == "__main__":
  hv.extension('bokeh')
  lre = LaserRateEquations(name='Gain Switching in SDLs')
  pn.serve(pn.Row(lre.param, lre.view), start=True, show=True, port=int(sys.argv[-1]), websocket_origin='lasereq.herokuapp.com')