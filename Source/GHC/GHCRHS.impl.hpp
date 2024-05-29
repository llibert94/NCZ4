/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(GHCRHS_HPP_)
#error "This file should only be included through GHCRHS.hpp"
#endif

#ifndef GHCRHS_IMPL_HPP_
#define GHCRHS_IMPL_HPP_

#include "DimensionDefinitions.hpp"
#include "GRInterval.hpp"
#include "VarsTools.hpp"

template <class gauge_t, class deriv_t>
inline GHCRHS<gauge_t, deriv_t>::GHCRHS(
    GHC_params_t<typename gauge_t::params_t> a_params, double a_dx,
    double a_sigma, double a_cosmological_constant)
    : m_params(a_params), m_gauge(a_params), m_sigma(a_sigma),
      m_cosmological_constant(a_cosmological_constant), m_deriv(a_dx)
{
}

template <class gauge_t, class deriv_t>
template <class data_t>
void GHCRHS<gauge_t, deriv_t>::compute(Cell<data_t> current_cell) const
{
    const auto vars = current_cell.template load_vars<Vars>();
    const auto d1 = m_deriv.template diff1<Vars>(current_cell);
    const auto d2 = m_deriv.template diff2<Diff2Vars>(current_cell);
    const auto advec =
        m_deriv.template advection<Vars>(current_cell, vars.shift);

    Vars<data_t> rhs;
    rhs_equation(rhs, vars, d1, d2, advec);

    m_deriv.add_dissipation(rhs, current_cell, m_sigma);

    current_cell.store_vars(rhs); // Write the rhs into the output FArrayBox
}

template <class gauge_t, class deriv_t>
template <class data_t, template <typename> class vars_t,
          template <typename> class diff2_vars_t>
void GHCRHS<gauge_t, deriv_t>::rhs_equation(
    vars_t<data_t> &rhs, const vars_t<data_t> &vars,
    const vars_t<Tensor<1, data_t>> &d1,
    const diff2_vars_t<Tensor<2, data_t>> &d2,
    const vars_t<data_t> &advec) const
{
    using namespace TensorAlgebra;

    auto g_UU = compute_inverse_sym(vars.g);
    auto chris = compute_christoffel(d1.g, g_UU);

    Tensor<1, data_t> Z;
    FOR(i) Z[i] = 0.5 * (vars.Gam[i] - chris.contracted[i]);

    auto ricci =
        GHCGeometry::compute_ricci_Z(vars, d1, d2, g_UU, chris, Z);

    data_t divshift = compute_trace(d1.shift);
    data_t Z_dot_d1lapse = compute_dot_product(Z, d1.lapse);

    Tensor<2, data_t> covd2lapse;
    FOR(k, l)
    {
        covd2lapse[k][l] = d2.lapse[k][l];
        FOR(m) { covd2lapse[k][l] -= chris.ULL[m][k][l] * d1.lapse[m]; }
    }

    data_t tr_covd2lapse = compute_trace(covd2lapse, g_UU);

    Tensor<2, data_t> K_UU = raise_all(vars.K, g_UU);

    data_t tr_K = compute_trace(vars.K, g_UU);
    // K^{ij} K_{ij}. - Note the abuse of the compute trace function.
    data_t tr_K2 = compute_trace(vars.K, K_UU);
    FOR(i, j)
    {
        rhs.g[i][j] = advec.g[i][j] - 2. * vars.lapse * vars.K[i][j];
        FOR(k)
        {
            rhs.g[i][j] +=
                vars.g[k][i] * d1.shift[k][j] + vars.g[k][j] * d1.shift[k][i];
        }
    }

    data_t kappa1_times_lapse;
    if (m_params.covariantZ4)
        kappa1_times_lapse = m_params.kappa1;
    else
        kappa1_times_lapse = m_params.kappa1 * vars.lapse;


    FOR(i, j)
    {
        rhs.K[i][j] = advec.K[i][j] - covd2lapse[i][j] + vars.lapse * ricci.LL[i][j] -
                      2. * vars.K[i][j] * vars.lapse * (tr_K - 2. * vars.Theta) -
		      kappa1_times_lapse * (1. + m_params.kappa2) * vars.Theta * vars.g[i][j];
        FOR(k)
        {
            rhs.K[i][j] +=
                vars.K[k][i] * d1.shift[k][j] + vars.K[k][j] * d1.shift[k][i];
            FOR(l)
            {
                rhs.K[i][j] -=
                    2. * vars.lapse * g_UU[k][l] * vars.K[i][k] * vars.K[l][j];
            }
        }
    }

    rhs.Theta =
         advec.Theta +
         0.5 * vars.lapse *
             (ricci.scalar - tr_K2 + tr_K * (tr_K - 2. * vars.Theta)) -
         0.5 * vars.Theta * kappa1_times_lapse *
             ((GR_SPACEDIM + 1) + m_params.kappa2 * (GR_SPACEDIM - 1)) -
         Z_dot_d1lapse;

     rhs.Theta += -vars.lapse * m_cosmological_constant;

    FOR(i)
    {
        rhs.Gam[i] = advec.Gam[i] - 2. * kappa1_times_lapse * Z[i];
        FOR(j)
        {
            rhs.Gam[i] +=
                g_UU[i][j] *
                    (2. * vars.lapse * d1.Theta[j] + (tr_K - 2. * vars.Theta) * d1.lapse[j]) -
                2. * K_UU[i][j] * d1.lapse[j];

            FOR(k)
            {
                rhs.Gam[i] +=
                    2. * vars.lapse * chris.ULL[i][j][k] * K_UU[j][k] +
                    g_UU[j][k] * d2.shift[i][j][k];
		FOR(l) rhs.Gam[i] += -vars.lapse * g_UU[j][k] * g_UU[i][l] * d1.K[j][k][l];
            }
        }
    }

    m_gauge.rhs_gauge(rhs, vars, d1, d2, advec);
}

#endif /* GHCRHS_IMPL_HPP_ */
