
/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef POINCAREADS_HPP_
#define POINCAREADS_HPP_

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

class PoincareAdS
{
  public:
    //! Struct for the params of the  BH
    struct params_t
    {
        double length = 1.0;                      //!<< length in AdS
	std::array<double, CH_SPACEDIM> center; //!< The center of the BH
    };

    template <class data_t> using Vars = GHCVars::VarsWithGauge<data_t>;
    template <class data_t>
    using Diff2Vars = GHCVars::Diff2VarsWithGauge<data_t>;

    const params_t m_params;
    const double m_dx;

    PoincareAdS(params_t a_params, double a_dx)
        : m_params(a_params), m_dx(a_dx)
    {
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

        // work out where we are on the grid
        const double z = coords.z;

	using namespace TensorAlgebra;
	const double z_reg = simd_max(1e-6, z);
        FOR(i, j)
        {
            g[i][j] = delta(i, j) * L * L / (z_reg * z_reg);
        }

        FOR(i, j, k)
        {
            dg[i][j][k] = -2. * delta(k, 2) * g[i][j] / z_reg;
        }
    }

    template <class data_t>
    void compute_d2g(Tensor<2, Tensor<1, data_t>> &dg,
		    	  Tensor<2, Tensor<1, Tensor<1, data_t>>> &d2g,
                          const Coordinates<data_t> &coords) const
    {
        const double L = m_params.length;

        // work out where we are on the grid
        const double z = coords.z;

        using namespace TensorAlgebra;
        const double z_reg = simd_max(1e-6, z);
        FOR(i, j, k, l)
        {
            d2g[i][j][k][l] = -3. * delta(l, 2) * dg[i][j][k] / z_reg;
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
        const double L = m_params.length;

        // work out where we are on the grid
        const double z = coords.z;

	using namespace TensorAlgebra;
        
	// populate ADM vars
	const double z_reg = simd_max(1e-6, z);
        vars.lapse = L / z_reg;
	FOR(i, j)
        {
            vars.h[i][j] = delta(i, j) * L * L / (z_reg * z_reg);
        }

        FOR(i)
        {
            vars.shift[i] = 0.;
        }

        // Calculate partial derivative of spatial metric
        FOR(i, j, k)
        {
            d1.h[i][j][k] = -2. * delta(k, 2) * vars.h[i][j] / z_reg;
        }

        FOR(i, j, k, l)
        {
            d2.h[i][j][k][l] = -3. * delta(l, 2) * d1.h[i][j][k] / z_reg;
        }

        // calculate derivs of lapse and shift
        FOR(i)
        {
            d1.lapse[i] = -delta(i, 2) * vars.lapse / z_reg;
        }

        FOR(i, j)
        {
            d2.lapse[i][j] = -2. * delta(j, 2) * d1.lapse[i] / z_reg;
        }

        // use the fact that shift^i = lapse^2 * shift_i
        FOR(i, j)
        {
            d1.shift[i][j] = 0.;
        }

        FOR(i, j, k)
        {
            d2.shift[i][j][k] = 0.;
        }

	data_t lapse_reg = simd_max(1e-6, vars.lapse);

	const auto g_UU = compute_inverse_sym(vars.h);
        // calculate the extrinsic curvature, using the fact that
        // 2 * lapse * K_ij = D_i \beta_j + D_j \beta_i - dgamma_ij dt
        // and dgamma_ij dt = 0 in chosen fixed gauge
        const auto chris = compute_christoffel(d1.h, g_UU);
        // FOR(i) vars.Gam[i] = chris.contracted[i];
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
};

#endif /* POINCAREADS_HPP_ */
