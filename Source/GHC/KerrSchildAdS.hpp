
/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef KERRSCHILDADS_HPP_
#define KERRSCHILDADS_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "GHCVars.hpp"
#include "Minkowski.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp" //This files needs NUM_VARS - total number of components
#include "simd.hpp"

//! Class which computes the initial conditions for a Kerr Schild BH
//! https://arxiv.org/pdf/gr-qc/9805023.pdf
//! https://arxiv.org/pdf/2011.07870.pdf

template <class background_t = Minkowski> class KerrSchildAdS
{
  public:
    //! Struct for the params of the  BH
    struct params_t
    {
        double length;
	double z0 = 1.0;                      //!<< The radius of the BH
        std::array<double, CH_SPACEDIM> center; //!< The center of the BH
    };

    template <class data_t> using Vars = GHCVars::VarsWithGauge<data_t>;
    template <class data_t>
    using Diff2Vars = GHCVars::Diff2VarsWithGauge<data_t>;

    const params_t m_params;
    const double m_dx;
    background_t m_background;

    KerrSchildAdS(params_t a_params, double a_dx, background_t a_background)
        : m_params(a_params), m_dx(a_dx), m_background(a_background)
    {
   	if (m_params.z0 > m_params.length)
        {
            MayDay::Error(
                "The radius of the black brane must be smaller than the AdS length");
        }
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        // get position and set vars
        const Coordinates<data_t> coords(current_cell, m_dx, m_params.center);
        Vars<data_t> metric_vars;
        Vars<Tensor<1, data_t>> d1;
        Diff2Vars<Tensor<2, data_t>> d2;
        Tensor<4, data_t> d_chris_ULL;
        Tensor<4, data_t> Riemann;

        compute_metric(metric_vars, d1, d2, d_chris_ULL, Riemann, coords);

        Tensor<2, data_t> bg_g;
        Tensor<2, Tensor<1, data_t>> bg_dg;
        m_background.compute_g_and_dg(bg_g, bg_dg, coords);

        FOR(i, j) {
	   metric_vars.h[i][j] -= bg_g[i][j];
	   FOR(k) d1.h[i][j][k] -= bg_dg[i][j][k];
	}

        FOR(i) metric_vars.B[i] = 0.;
        // Populate the variables on the grid
        // NB We stil need to set Gamma^i which is NON ZERO
        // but we do this via a separate class/compute function
        // as we need the gradients of the metric which are not yet available
        current_cell.store_vars(metric_vars);
    }

    template <class data_t>
    void compute_g_and_dg(Tensor<2, data_t> &g,
                          Tensor<2, Tensor<1, data_t>> &dg,
                          const Coordinates<data_t> &coords) const
    {
        const double L = m_params.length;
	const double z0 = m_params.radius;

        // work out where we are on the grid
	const double z = coords.z;
        const double z_reg = simd_max(1e-6, z);

        // find the H and el quantities (el decomposed into space and time)
        data_t fac = L * L / (z_reg * z_reg);
	data_t H = pow(z / z0, GR_SPACEDIM + 1.);

        const Tensor<1, data_t> el = {0., 0., 1.};
	Tensor<1, data_t> dHdx;

        using namespace TensorAlgebra;

        FOR(i) dHdx[i] = (GR_SPACEDIM + 1.) / z_reg * H * delta(i, 2);
        
        FOR(i, j)
        {
            g[i][j] = fac * (delta(i, j) + H * el[i] * el[j]);
        }

        FOR(i, j, k)
        {
            dg[i][j][k] = -2. / z_reg * delta(k, 2) * g[i][j] + 
                	  el[i] * el[j] * dHdx[k];
        }
    }

    // Kerr Schild solution
    template <class data_t, template <typename> class vars_t,
              template <typename> class diff2_vars_t>
    void compute_metric(vars_t<data_t> &vars, vars_t<Tensor<1, data_t>> &d1,
                        diff2_vars_t<Tensor<2, data_t>> &d2,
                        Tensor<4, data_t> &d_chris_ULL,
                        Tensor<4, data_t> &Riemann,
                        const Coordinates<data_t> &coords) const
    {
        // AdS and black hole params
	const double L = m_params.length;
	const double z0 = m_params.z0;

        // work out where we are on the grid
        const double z = coords.z;

        // find the H and el quantities (el decomposed into space and time)
        const Tensor<1, data_t> el = {0., 0., 1.};
        const data_t el_t = 1.0;

	const double z_reg = simd_max(1e-6, z);

        using namespace TensorAlgebra;

        data_t fac = L * L / (z_reg * z_reg);
	data_t H = pow(z / z0, GR_SPACEDIM + 1.);
	Tensor<1, data_t> dHdx;
	Tensor<2, data_t> d2Hdx2;
        FOR(i)
           dHdx[i] = (GR_SPACEDIM + 1.) / z_reg * H * delta(i, 2);

        FOR(i, j)
           d2Hdx2[i][j] = GR_SPACEDIM / z_reg * delta(j, 2) * dHdx[i];

        // populate ADM vars
        vars.lapse = L / z_reg * pow(1.0 - H * el_t * el_t, -0.5);
	
	using namespace TensorAlgebra;

        FOR(i, j) vars.h[i][j] = fac * (delta(i, j) + H * el[i] * el[j] - delta(i, 2) * delta(j, 2));

        const auto g_UU = compute_inverse_sym(vars.h);

        FOR(i)
        {
            vars.shift[i] = 0.;
            FOR(j) { vars.shift[i] += g_UU[i][j] /* fac */ * H * el[j] * el_t; }
        }

        // Calculate partial derivative of spatial metric
        FOR(i, j, k)
        {
            d1.h[i][j][k] = -2. / z_reg * delta(k, 2) * vars.h[i][j] + fac * el[i] * el[j] * dHdx[k];
        }

        FOR(i, j, k, l)
        {
            d2.h[i][j][k][l] = -2. / z_reg* delta(k, 2) * (2. * d1.h[i][j][l] +  delta(l, 2) / z_reg * vars.h[i][j]) + 
		    fac * el[i] * el[j] * d2Hdx2[k][l];
        }

        // calculate derivs of lapse and shift
        FOR(i)
        {
	    d1.lapse[i] = vars.lapse * (0.5 * el_t * el_t * dHdx[i] / (1.0 - H * el_t * el_t) - delta(i, 2) / z_reg);
        }

        FOR(i, j)
        {
	    d2.lapse[i][j] =
                0.5 * el_t * el_t * (d2Hdx2[i][j] * vars.lapse + dHdx[i] * d1.lapse[j]
                                + (dHdx[i] * vars.lapse * dHdx[j] * el_t * el_t) / (1.0 - H * el_t * el_t))
                                / (1.0 - H * el_t * el_t)
                - delta(i, 2) / z_reg * (d1.lapse[j] - vars.lapse / z_reg * delta(j, 2));
        }

        // use the fact that shift^i = lapse^2 * shift_i
        FOR(i, j)
        {
            d1.shift[i][j] =
                2.0 * el_t /* fac */ * (dHdx[j] /*- H / z_reg * delta(j, 2)*/) * pow(vars.lapse, 2.0) * el[i] +
                4.0 * el_t /* fac */ * H * vars.lapse * d1.lapse[j] * el[i];
        }

        FOR(i, j, k)
        {
            d2.shift[i][j][k] =
		/*-2.0 * el_t / z_reg * delta(k, 2) * fac * (dHdx[j] - H / z_reg * delta(j, 2)) * 
				pow(vars.lapse, 2.0) * el[i] +*/
                2.0 * el_t /* fac */ * (d2Hdx2[j][k] /*- delta(j, 2) / z_reg *
				(dHdx[k] - H / z_reg * delta(k, 2))*/) * pow(vars.lapse, 2.0) * el[i] /*-
                4.0 * el_t / z_reg * delta(k, 2) * fac * H * vars.lapse * d1.lapse[j] * el[i]*/ +
		4.0 * el_t /* fac */ * (dHdx[j] /*- H / z_reg * delta(j, 2)*/) * vars.lapse * d1.lapse[k] * el[i] +
                4.0 * el_t /* fac */ * dHdx[k] * vars.lapse * d1.lapse[j] * el[i] +
                4.0 * el_t /* fac */ * H * d1.lapse[k] * d1.lapse[j] * el[i] +
                4.0 * el_t /* fac */ * H * vars.lapse * d2.lapse[j][k] * el[i];
        }

        // calculate the extrinsic curvature, using the fact that
        // 2 * lapse * K_ij = D_i \beta_j + D_j \beta_i - dgamma_ij dt
        // and dgamma_ij dt = 0 in chosen fixed gauge
        const auto chris = compute_christoffel(d1.h, g_UU);
        // FOR(i) vars.Gam[i] = chris.contracted[i];
        const data_t lapse_reg = simd_max(1e-6, vars.lapse);
	FOR(i, j)
        {
            vars.K[i][j] = 0.0;
            FOR(k)
            {
                vars.K[i][j] += vars.h[k][j] * d1.shift[k][i] +
                                vars.h[k][i] * d1.shift[k][j] +
                                (d1.h[k][i][j] + d1.h[k][j][i]) * vars.shift[k];
                FOR(m)
                {
                    vars.K[i][j] += -2.0 * chris.ULL[k][i][j] * vars.h[k][m] *
                                    vars.shift[m];
                }
            }
            vars.K[i][j] *= 0.5 / lapse_reg;
        }
        vars.Pi = -compute_trace(vars.K, g_UU);

        FOR(i, j, k, l)
        {
            d_chris_ULL[i][j][k][l] = 0.;
            FOR(m)
            {
                d_chris_ULL[i][j][k][l] +=
                    0.5 * g_UU[i][m] *
                    (d2.h[k][m][j][l] + d2.h[j][m][k][l] - d2.h[j][k][m][l]);
                FOR(p, q)
                {
                    d_chris_ULL[i][j][k][l] -= g_UU[i][p] * g_UU[m][q] *
                                               d1.h[p][q][l] *
                                               chris.LLL[m][j][k];
                }
            }
        }

        FOR(i, j, k, l)
        {
            Riemann[i][j][k][l] =
                d_chris_ULL[i][l][j][k] - d_chris_ULL[i][k][j][l];
            FOR(m)
            {
                Riemann[i][j][k][l] += chris.ULL[i][k][m] * chris.ULL[m][l][j] -
                                       chris.ULL[i][l][m] * chris.ULL[m][k][j];
            }
        }

        FOR(i, j, k)
        {
            d1.K[i][j][k] = 0.0;
            FOR(l)
            {
                d1.K[i][j][k] +=
                    d1.h[l][j][k] * d1.shift[l][i] +
                    vars.h[l][j] * d2.shift[l][i][k] +
                    d1.h[l][i][k] * d1.shift[l][j] +
                    vars.h[l][i] * d2.shift[l][j][k] +
                    (d2.h[l][i][j][k] + d2.h[l][j][i][k]) * vars.shift[l] +
                    (d1.h[l][i][j] + d1.h[l][j][i]) * d1.shift[l][k];
                FOR(m)
                {
                    d1.K[i][j][k] += -2.0 * chris.ULL[l][i][j] *
                                         (d1.h[l][m][k] * vars.shift[m] +
                                          vars.h[l][m] * d1.shift[m][k]) -
                                     2.0 * d_chris_ULL[l][i][j][k] *
                                         vars.h[l][m] * vars.shift[m];
                }
            }
            d1.K[i][j][k] *= 0.5 / lapse_reg;
            d1.K[i][j][k] -= vars.K[i][j] * d1.lapse[k] / lapse_reg;
        }
    }

  public:
    // used to decide when to excise - ie when within the horizon of the BH
    // note that this is not templated over data_t
    bool check_if_excised(const Coordinates<double> &coords) const
    {
	bool is_excised = false;
        // value less than 1 indicates we are within the horizon
        if (coords.z < m_params.z0)
        {
            is_excised = true;
        }
        return is_excised;
    }
};

#endif /* KERRSCHILDADS_HPP_ */
