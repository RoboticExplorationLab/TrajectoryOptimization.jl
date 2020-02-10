#pragma once

/* returns fy according to the fiala tire model */
double fiala(double alpha, double Ca, double mu, double Fx, double Fz);

/* returns max fy, derating by fx */
double calc_fy_max(double mu, double fx_N, double fz_N);

