/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(CONFORMALDIAGNOSTICS_HPP_)
#error "This file should only be included through ConformalDiagnostics.hpp"
#endif

#ifndef CONFORMALDIAGNOSTICS_IMPL_HPP_
#define CONFORMALDIAGNOSTICS_IMPL_HPP_

#include "DimensionDefinitions.hpp"
#include "GRInterval.hpp"
#include "VarsTools.hpp"

template <class background_t>
inline ConformalDiagnostics<background_t>::ConformalDiagnostics(
    double dx, const std::array<double, CH_SPACEDIM> a_center,
    background_t a_background, const Interval &a_c_gs, int a_c_chi,
    const Interval &a_c_CGams, const Interval &a_c_Zs)
    : m_deriv(dx), m_center(a_center), m_background(a_background),
      m_c_gs(a_c_gs), m_c_chi(a_c_chi), m_c_CGams(a_c_CGams), m_c_Zs(a_c_Zs)
{
}

template <class background_t>
template <class data_t>
void ConformalDiagnostics<background_t>::compute(
    Cell<data_t> current_cell) const
{
    const auto vars = current_cell.template load_vars<MetricVars>();
    const auto d1 = m_deriv.template diff1<MetricVars>(current_cell);
    const Coordinates<data_t> coords(current_cell, m_deriv.m_dx, m_center);

    Vars<data_t> out = conformal_diagnostics_equations(vars, d1, coords);

    // Write the rhs into the output FArrayBox
    current_cell.store_vars(out);
}

template <class background_t>
template <class data_t, template <typename> class vars_t>
ConformalDiagnostics<background_t>::Vars<data_t>
ConformalDiagnostics<background_t>::conformal_diagnostics_equations(
    const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
    const Coordinates<data_t> &coords) const
{
    Vars<data_t> out;

    if (m_c_gs.size() > 0 || m_c_chi >= 0 || m_c_CGams.size() > 0 ||
        m_c_Zs.size() > 0)
    {
        Tensor<2, data_t> bg_g;
        Tensor<2, Tensor<1, data_t>> bg_dg;
        m_background.compute_g_and_dg(bg_g, bg_dg, coords);

        FOR(i, j) { out.g[i][j] = vars.h[i][j] + bg_g[i][j]; }

        data_t det_g = TensorAlgebra::compute_determinant_sym(out.g);
        det_g = sqrt(det_g * det_g);
        out.chi = pow(det_g, -1. / (double)GR_SPACEDIM);

        if (m_c_CGams.size() > 0 || m_c_Zs.size() > 0)
        {
            auto bg_h_UU = TensorAlgebra::compute_inverse_sym(bg_g);
            auto bg_chris = TensorAlgebra::compute_christoffel(bg_dg, bg_h_UU);

            data_t chi_regularised = simd_max(1e-6, out.chi);
            auto g_UU = TensorAlgebra::compute_inverse_sym(out.g);
            auto diff_chris = TensorAlgebra::compute_christoffel(d1.h, g_UU);

            FOR(i, j, k, l, m)
            {
                diff_chris.contracted[i] -= bg_chris.ULL[l][j][k] *
                                            vars.h[m][l] * g_UU[i][m] *
                                            g_UU[j][k];
            }

            FOR(i)
            {
                out.CGam[i] = diff_chris.contracted[i] / chi_regularised;
                out.Z[i] = 0.5 * (vars.Gam[i] - diff_chris.contracted[i]);
            }
            FOR(i, j, k, l)
            {
                out.CGam[i] += g_UU[i][j] * g_UU[k][l] * d1.h[k][l][j] /
                               (6. * chi_regularised);
            }
        }
    }
    return out;
}

#endif /* CONFORMALDIAGNOSTICS_IMPL_HPP_ */
