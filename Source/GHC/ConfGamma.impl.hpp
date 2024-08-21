/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(CONFGAMMA_HPP_)
#error "This file should only be included through ConfGamma.hpp"
#endif

#ifndef CONFGAMMA_IMPL_HPP_
#define CONFGAMMA_IMPL_HPP_

#include "DimensionDefinitions.hpp"
#include "GRInterval.hpp"
#include "VarsTools.hpp"

inline ConfGamma::ConfGamma(
    double dx, const Interval &a_c_CGams)
    : m_deriv(dx), m_c_CGams(a_c_CGams)
{
}

template <class data_t>
void ConfGamma::compute(Cell<data_t> current_cell) const
{
    const auto vars = current_cell.template load_vars<MetricVars>();
    const auto d1 = m_deriv.template diff1<MetricVars>(current_cell);

    const auto g_UU = TensorAlgebra::compute_inverse_sym(vars.g);
    const auto chris = TensorAlgebra::compute_christoffel(d1.g, g_UU);

    Vars<data_t> out = conf_gamma_equations(vars, d1, g_UU, chris);

    // Write the rhs into the output FArrayBox
    current_cell.store_vars(out);
}

template <class data_t, template <typename> class vars_t>
ConfGamma::Vars<data_t> ConfGamma::conf_gamma_equations(
    const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
    const Tensor<2, data_t> &g_UU, const chris_t<data_t> &chris) const
{
    Vars<data_t> out;
   
    if (m_c_CGams.size() > 0) {
	
    	data_t det_g = TensorAlgebra::compute_determinant_sym(vars.g);
        out.chi = pow(det_g, -1. / (double)GR_SPACEDIM);
	data_t chi_regularised = simd_max(1e-4, out.chi);

        FOR(i) { 
	   out.CGam[i] = chris.contracted[i] / chi_regularised;
	   out.Z[i] = vars.Gam[i] - chris.contracted[i]; 
	}
    	FOR(i, j, k,l)
    	{	
           out.CGam[i] += g_UU[i][j] * g_UU[k][l] * d1.g[k][l][j] / (6. * chi_regularised); 
    	}
    }
    return out;
}

#endif /* CONFGAMMA_IMPL_HPP_ */
