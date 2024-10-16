/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GAMMACALCULATOR_HPP_
#define GAMMACALCULATOR_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GRInterval.hpp"
#include "KerrSchild.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp" //This files needs NUM_VARS - total number of components
#include "VarsTools.hpp"
#include "simd.hpp"

class GammaCalculator
{
    // Only variables needed are metric
    template <class data_t> struct Vars
    {
        Tensor<2, data_t> g;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
        {
            VarsTools::define_symmetric_enum_mapping(
                mapping_function, GRInterval<c_g11, c_g33>(), g);
        }
    };

  protected:
    const FourthOrderDerivatives
        m_deriv; //!< An object for calculating derivatives of the variables
    const std::array<double, CH_SPACEDIM> m_center;
    const bool m_kerr_bg;

  public:
    GammaCalculator(double a_dx, const std::array<double, CH_SPACEDIM> a_center,
		    bool a_kerr_bg) : 
	    m_deriv(a_dx), m_center(a_center), m_kerr_bg(a_kerr_bg) {}

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        // copy data from chombo gridpoint into local variables, and calc 1st
        // derivs
        const auto vars = current_cell.template load_vars<Vars>();
        const auto d1 = m_deriv.template diff1<Vars>(current_cell);

	Coordinates<data_t> coords{current_cell, this->m_deriv.m_dx, m_center};

        using namespace TensorAlgebra;
        const auto g_UU = compute_inverse_sym(vars.g);
        const auto chris = compute_christoffel(d1.g, g_UU);

	Tensor<2, data_t> bg_g;
    	Tensor<2, Tensor<1, data_t>> bg_dg;
    	KerrSchild::params_t kerr_params;
    	kerr_params.mass = 1.0;
    	kerr_params.spin = 0.0;
    	kerr_params.center = m_center;
    	KerrSchild kerr_schild(kerr_params, this->m_deriv.m_dx);
    	kerr_schild.compute_g_and_dg(bg_g, bg_dg, coords);

    	auto gbar_UU = compute_inverse_sym(bg_g);
    	auto bg_chris = compute_christoffel(bg_dg, gbar_UU);

	Tensor<1, data_t> gamma = chris.contracted;

	if (m_kerr_bg) {
            FOR(i, j, k) gamma[i] += -g_UU[j][k] * bg_chris.ULL[i][j][k];
    	}

        // assign values of Gamma^k = g_UU^ij * \tilde{Gamma}^k_ij in the output
        // FArrayBox
        current_cell.store_vars(gamma,
                                GRInterval<c_Gam1, c_Gam3>());
	/*current_cell.store_vars(chris.contracted,
                                GRInterval<c_shift1, c_shift3>());*/
    }
};

#endif /* GAMMACALCULATOR_HPP_ */
