/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(NEWCONSTRAINTS_HPP_)
#error "This file should only be included through NewConstraints.hpp"
#endif

#ifndef NEWCONSTRAINTS_IMPL_HPP_
#define NEWCONSTRAINTS_IMPL_HPP_

#include "DimensionDefinitions.hpp"
#include "GRInterval.hpp"
#include "VarsTools.hpp"

template <class background_t>
inline Constraints<background_t>::Constraints(
    double dx, const std::array<double, CH_SPACEDIM> a_center,
    background_t a_background, int a_c_Ham, const Interval &a_c_Moms,
    int a_c_Ham_abs_terms /*defaulted*/,
    const Interval &a_c_Moms_abs_terms /*defaulted*/,
    double cosmological_constant /*defaulted*/)
    : m_deriv(dx), m_center(a_center), m_background(a_background),
      m_c_Ham(a_c_Ham), m_c_Moms(a_c_Moms),
      m_c_Ham_abs_terms(a_c_Ham_abs_terms),
      m_c_Moms_abs_terms(a_c_Moms_abs_terms),
      m_cosmological_constant(cosmological_constant)
{
}

template <class background_t>
template <class data_t>
void Constraints<background_t>::compute(Cell<data_t> current_cell) const
{
    const auto vars = current_cell.template load_vars<MetricVars>();
    const auto d1 = m_deriv.template diff1<MetricVars>(current_cell);
    const auto d2 = m_deriv.template diff2<Diff2Vars>(current_cell);

    Coordinates<data_t> coords{current_cell, this->m_deriv.m_dx, m_center};

    Vars<data_t> out = constraint_equations(vars, d1, d2, coords);

    store_vars(out, current_cell);
}

template <class background_t>
template <class data_t, template <typename> class vars_t,
          template <typename> class diff2_vars_t>
Constraints<background_t>::Vars<data_t>
Constraints<background_t>::constraint_equations(
    const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
    const diff2_vars_t<Tensor<2, data_t>> &d2,
    const Coordinates<data_t> &coords) const
{
    Vars<data_t> out;

    vars_t<data_t> bg_vars;
    vars_t<Tensor<1, data_t>> bg_d1;
    diff2_vars_t<Tensor<2, data_t>> bg_d2;
    Tensor<4, data_t> d_bg_chris_ULL;
    Tensor<4, data_t> bg_Riemann;
    m_background.compute_metric(bg_vars, bg_d1, bg_d2, d_bg_chris_ULL,
                                bg_Riemann, coords);

    using namespace TensorAlgebra;
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

    if (m_c_Ham >= 0 || m_c_Ham_abs_terms >= 0)
    {

        auto ricci =
            GHCGeometry::compute_ricci(vars, d1, d2, g_UU, phys_g, diff_chris,
                                       bg_chris, d_bg_chris_ULL, bg_Riemann);

        Tensor<2, data_t> K_UU = raise_all(vars.K, g_UU);
        data_t tr_K = compute_trace(vars.K, g_UU);
        data_t tr_K2 = compute_trace(vars.K, K_UU);

        out.Ham = ricci.scalar + tr_K * tr_K - tr_K2;
        out.Ham -= 2 * m_cosmological_constant;

        out.Ham_abs_terms =
            abs(ricci.scalar) + abs(tr_K2) + abs(tr_K * tr_K / GR_SPACEDIM);
        out.Ham_abs_terms += 2.0 * abs(m_cosmological_constant);
    }

    if (m_c_Moms.size() > 0 || m_c_Moms_abs_terms.size() > 0)
    {
        Tensor<2, data_t> covd_K[CH_SPACEDIM];
        FOR(i, j, k)
        {
            covd_K[i][j][k] = d1.K[j][k][i];
            FOR(l)
            {
                covd_K[i][j][k] +=
                    -(diff_chris.ULL[l][i][j] + bg_chris.ULL[l][i][j]) *
                        vars.K[l][k] -
                    (diff_chris.ULL[l][i][k] + bg_chris.ULL[l][i][k]) *
                        vars.K[l][j];
            }
        }
        FOR(i)
        {
            out.Mom[i] = 0.;
            FOR(j, k) out.Mom[i] += g_UU[j][k] * covd_K[i][j][k];
            out.Mom_abs_terms[i] = abs(out.Mom[i]);
        }
        Tensor<1, data_t> covd_K_term = {0.};
        FOR(i, j, k) covd_K_term[i] += g_UU[j][k] * covd_K[j][k][i];
        FOR(i)
        {
            out.Mom[i] -= covd_K_term[i];
            out.Mom_abs_terms[i] += abs(covd_K_term[i]);
        }
    }
    return out;
}

template <class background_t>
template <class data_t>
void Constraints<background_t>::store_vars(Vars<data_t> &out,
                                           Cell<data_t> &current_cell) const
{
    if (m_c_Ham >= 0)
        current_cell.store_vars(out.Ham, m_c_Ham);
    if (m_c_Ham_abs_terms >= 0)
        current_cell.store_vars(out.Ham_abs_terms, m_c_Ham_abs_terms);
    if (m_c_Moms.size() == GR_SPACEDIM)
    {
        FOR(i)
        {
            int ivar = m_c_Moms.begin() + i;
            current_cell.store_vars(out.Mom[i], ivar);
        }
    }
    else if (m_c_Moms.size() == 1)
    {
        data_t Mom_sq = 0.0;
        FOR(i) { Mom_sq += out.Mom[i] * out.Mom[i]; }
        data_t Mom = sqrt(Mom_sq);
        current_cell.store_vars(Mom, m_c_Moms.begin());
    }
    if (m_c_Moms_abs_terms.size() == GR_SPACEDIM)
    {
        FOR(i)
        {
            int ivar = m_c_Moms_abs_terms.begin() + i;
            current_cell.store_vars(out.Mom_abs_terms[i], ivar);
        }
    }
    else if (m_c_Moms_abs_terms.size() == 1)
    {
        data_t Mom_abs_terms_sq = 0.0;
        FOR(i)
        {
            Mom_abs_terms_sq += out.Mom_abs_terms[i] * out.Mom_abs_terms[i];
        }
        data_t Mom_abs_terms = sqrt(Mom_abs_terms_sq);
        current_cell.store_vars(Mom_abs_terms, m_c_Moms_abs_terms.begin());
    }
}

#endif /* NEWCONSTRAINTS_IMPL_HPP_ */
