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
#include "KerrSchild.hpp"
#include "VarsTools.hpp"

template <class gauge_t, class deriv_t, class background_t>
inline GHCRHS<gauge_t, deriv_t, background_t>::GHCRHS(
    GHC_params_t<typename gauge_t::params_t> a_params, double a_dx,
    double a_sigma, const std::array<double, CH_SPACEDIM> a_center,
    background_t a_background, double a_cosmological_constant)
    : m_params(a_params), m_gauge(a_params, a_background), m_sigma(a_sigma),
      m_cosmological_constant(a_cosmological_constant), m_deriv(a_dx),
      m_center(a_center), m_background(a_background)
{
}

template <class gauge_t, class deriv_t, class background_t>
template <class data_t>
void GHCRHS<gauge_t, deriv_t, background_t>::compute(
    Cell<data_t> current_cell) const
{
    const auto vars = current_cell.template load_vars<Vars>();
    const auto d1 = m_deriv.template diff1<Vars>(current_cell);
    const auto d2 = m_deriv.template diff2<Diff2Vars>(current_cell);
    const auto advec =
        m_deriv.template advection<Vars>(current_cell, vars.shift);

    Coordinates<data_t> coords{current_cell, this->m_deriv.m_dx, m_center};

    Vars<data_t> rhs;
    rhs_equation(rhs, vars, d1, d2, advec, coords);

    m_deriv.add_dissipation(rhs, current_cell, m_sigma);

    current_cell.store_vars(rhs); // Write the rhs into the output FArrayBox
}

template <class gauge_t, class deriv_t, class background_t>
template <class data_t, template <typename> class vars_t,
          template <typename> class diff2_vars_t>
void GHCRHS<gauge_t, deriv_t, background_t>::rhs_equation(
    vars_t<data_t> &rhs, const vars_t<data_t> &vars,
    const vars_t<Tensor<1, data_t>> &d1,
    const diff2_vars_t<Tensor<2, data_t>> &d2, const vars_t<data_t> &advec,
    const Coordinates<data_t> &coords) const
{
    using namespace TensorAlgebra;

    Vars<data_t> bg_vars;
    Vars<Tensor<1, data_t>> bg_d1;
    diff2_vars_t<Tensor<2, data_t>> bg_d2;
    Tensor<4, data_t> d_bg_chris_ULL;
    Tensor<4, data_t> bg_Riemann;
    m_background.compute_metric(bg_vars, bg_d1, bg_d2, d_bg_chris_ULL,
                                bg_Riemann, coords);

    auto bg_h_UU = compute_inverse_sym(bg_vars.h);
    auto bg_chris = compute_christoffel(bg_d1.h, bg_h_UU);

    Tensor<2, data_t> phys_g;
    FOR(i, j) { phys_g[i][j] = vars.h[i][j] + bg_vars.h[i][j]; }

    auto g_UU = compute_inverse_sym(phys_g);
    auto diff_chris = compute_christoffel(d1.h, g_UU);

    // this will be needed (as all the places where bg_chris appears) until we
    // use directly the bg covariant derivative of the vars rather than their
    // partial derivative

    FOR(i, j, k, l)
    {
        diff_chris.LLL[i][j][k] -= bg_chris.ULL[l][j][k] * vars.h[i][l];
        FOR(m)
        {
            diff_chris.ULL[i][j][k] -=
                bg_chris.ULL[l][j][k] * vars.h[m][l] * g_UU[i][m];
            diff_chris.contracted[i] -=
                bg_chris.ULL[l][j][k] * vars.h[m][l] * g_UU[i][m] * g_UU[j][k];
        }
    }

    Tensor<1, data_t> Z;
    FOR(i) Z[i] = 0.5 * (vars.Gam[i] - diff_chris.contracted[i]);

    auto ricci =
        GHCGeometry::compute_ricci_Z(vars, d1, d2, g_UU, phys_g, diff_chris, Z,
                                     bg_chris, d_bg_chris_ULL, bg_Riemann);

    data_t divshift = compute_trace(d1.shift);
    data_t Z_dot_d1lapse = compute_dot_product(Z, d1.lapse);

    Tensor<2, data_t> covd2lapse;
    FOR(k, l)
    {
        covd2lapse[k][l] = d2.lapse[k][l];
        FOR(m)
        covd2lapse[k][l] -=
            (diff_chris.ULL[m][k][l] + bg_chris.ULL[m][k][l]) * d1.lapse[m];
    }

    data_t tr_covd2lapse = compute_trace(covd2lapse, g_UU);

    Tensor<2, data_t> K_UU = raise_all(vars.K, g_UU);

    data_t tr_K = compute_trace(vars.K, g_UU);
    // K^{ij} K_{ij}. - Note the abuse of the compute trace function.
    data_t tr_K2 = compute_trace(vars.K, K_UU);
    FOR(i, j)
    {
        rhs.h[i][j] = advec.h[i][j] - 2. * vars.lapse * vars.K[i][j];
        FOR(k)
        {
            rhs.h[i][j] += phys_g[k][i] * d1.shift[k][j] +
                           phys_g[k][j] * d1.shift[k][i] +
                           bg_d1.h[i][j][k] * vars.shift[k];
        }
    }

    data_t Theta = 0.5 * (vars.Pi + tr_K);

    data_t kappa1_times_lapse;
    if (m_params.covariantZ4)
        kappa1_times_lapse = m_params.kappa1;
    else
        kappa1_times_lapse = m_params.kappa1 * vars.lapse;

    FOR(i, j)
    {
        rhs.K[i][j] =
            advec.K[i][j] - covd2lapse[i][j] + vars.lapse * ricci.LL[i][j] -
            vars.K[i][j] * vars.lapse * vars.Pi -
            kappa1_times_lapse * (1. + m_params.kappa2) * Theta * phys_g[i][j];
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

    rhs.Pi = advec.Pi - vars.lapse * tr_K2 + tr_covd2lapse -
             Theta * kappa1_times_lapse * (1. - m_params.kappa2) -
             2. * Z_dot_d1lapse;

    rhs.Pi += -2. * vars.lapse * m_cosmological_constant;

    FOR(i)
    {
        rhs.Gam[i] = advec.Gam[i] - 2. * kappa1_times_lapse * Z[i];
        FOR(j)
        {
            rhs.Gam[i] +=
                g_UU[i][j] * (vars.lapse * d1.Pi[j] - vars.Pi * d1.lapse[j]) -
                2. * K_UU[i][j] * d1.lapse[j] - vars.Gam[j] * d1.shift[i][j];

            FOR(k)
            {
                rhs.Gam[i] +=
                    2. * vars.lapse * diff_chris.ULL[i][j][k] * K_UU[j][k] +
                    g_UU[j][k] * d2.shift[i][j][k];
                rhs.Gam[i] += -g_UU[j][k] * bg_d2.shift[i][j][k];
                FOR(l)
                {
                    rhs.Gam[i] +=
                        g_UU[j][k] * (bg_chris.ULL[i][j][l] *
                                          (d1.shift[l][k] - bg_d1.shift[l][k]) -
                                      bg_chris.ULL[l][j][k] *
                                          (d1.shift[i][l] - bg_d1.shift[i][l]));
                    rhs.Gam[i] +=
                        g_UU[j][k] * (bg_chris.ULL[i][k][l] *
                                          (d1.shift[l][j] - bg_d1.shift[l][j]) +
                                      d_bg_chris_ULL[i][k][l][j] *
                                          (vars.shift[l] - bg_vars.shift[l]));
                    rhs.Gam[i] += bg_h_UU[i][j] * g_UU[k][l] *
                                  (2. * bg_d1.lapse[k] * bg_vars.K[j][l] +
                                   2. * bg_vars.lapse * bg_d1.K[j][l][k] -
                                   bg_d1.lapse[j] * bg_vars.K[k][l] -
                                   bg_vars.lapse * bg_d1.K[k][l][j]);
                    FOR(m)
                    {
                        rhs.Gam[i] +=
                            g_UU[j][k] * (vars.shift[l] - bg_vars.shift[l]) *
                            (bg_chris.ULL[m][k][l] * bg_chris.ULL[i][m][j] -
                             bg_chris.ULL[i][m][l] * bg_chris.ULL[m][j][k]);
                        rhs.Gam[i] -=
                            bg_h_UU[i][j] * g_UU[k][l] * bg_vars.lapse *
                            (2. * bg_vars.K[j][m] * bg_chris.ULL[m][k][l] +
                             bg_vars.K[m][l] * bg_chris.ULL[m][j][k] -
                             bg_vars.K[k][m] * bg_chris.ULL[m][j][l]);
                        FOR(n)
                        rhs.Gam[i] -= bg_h_UU[i][j] * g_UU[k][l] *
                                      bg_vars.h[m][n] * bg_Riemann[n][k][l][j] *
                                      (vars.shift[m] - bg_vars.shift[m]);
                    }
                }
            }
        }
    }

    m_gauge.rhs_gauge(rhs, vars, d1, d2, advec, coords);
}

#endif /* GHCRHS_IMPL_HPP_ */
