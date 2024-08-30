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

inline Constraints::Constraints(
    double dx, int a_c_Ham, const Interval &a_c_Moms,
    int a_c_Ham_abs_terms /*defaulted*/,
    const Interval &a_c_Moms_abs_terms /*defaulted*/,
    double cosmological_constant /*defaulted*/)
    : m_deriv(dx), m_c_Ham(a_c_Ham), m_c_Moms(a_c_Moms),
      m_c_Ham_abs_terms(a_c_Ham_abs_terms),
      m_c_Moms_abs_terms(a_c_Moms_abs_terms),
      m_cosmological_constant(cosmological_constant)
{
}

template <class data_t>
void Constraints::compute(Cell<data_t> current_cell) const
{
    const auto vars = current_cell.template load_vars<MetricVars>();
    const auto d1 = m_deriv.template diff1<MetricVars>(current_cell);
    const auto d2 = m_deriv.template diff2<Diff2Vars>(current_cell);

    const auto g_UU = TensorAlgebra::compute_inverse_sym(vars.g);
    const auto chris = TensorAlgebra::compute_christoffel(d1.g, g_UU);

    Vars<data_t> out = constraint_equations(vars, d1, d2, g_UU, chris);

    store_vars(out, current_cell);
}

template <class data_t, template <typename> class vars_t,
          template <typename> class diff2_vars_t>
Constraints::Vars<data_t> Constraints::constraint_equations(
    const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
    const diff2_vars_t<Tensor<2, data_t>> &d2, const Tensor<2, data_t> &g_UU,
    const chris_t<data_t> &chris) const
{
    Vars<data_t> out;

    if (m_c_Ham >= 0 || m_c_Ham_abs_terms >= 0)
    {
        auto ricci = GHCGeometry::compute_ricci(vars, d1, d2, g_UU, chris);

        Tensor<2, data_t> K_UU = TensorAlgebra::raise_all(vars.K, g_UU);
	data_t tr_K = TensorAlgebra::compute_trace(vars.K, g_UU);
        data_t tr_K2 = TensorAlgebra::compute_trace(vars.K, K_UU);

        out.Ham = ricci.scalar + tr_K * tr_K - tr_K2;
        out.Ham -= 2 * m_cosmological_constant;

        out.Ham_abs_terms =
            abs(ricci.scalar) + abs(tr_K2) +
            abs(tr_K * tr_K / GR_SPACEDIM);
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
                covd_K[i][j][k] += -chris.ULL[l][i][j] * vars.K[l][k] -
                                   chris.ULL[l][i][k] * vars.K[l][j];
            }
        }
        FOR(i)
        {
            out.Mom[i] = 0.;
	    FOR(j,k) out.Mom[i] += g_UU[j][k] * covd_K[i][j][k];
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

template <class data_t>
void Constraints::store_vars(Vars<data_t> &out,
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
