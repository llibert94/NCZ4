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

  public:
    GammaCalculator(double a_dx) : m_deriv(a_dx) {}

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        // copy data from chombo gridpoint into local variables, and calc 1st
        // derivs
        const auto vars = current_cell.template load_vars<Vars>();
        const auto d1 = m_deriv.template diff1<Vars>(current_cell);

        using namespace TensorAlgebra;
        const auto g_UU = compute_inverse_sym(vars.g);
        const auto chris = compute_christoffel(d1.g, g_UU);

        // assign values of Gamma^k = g_UU^ij * \tilde{Gamma}^k_ij in the output
        // FArrayBox
        current_cell.store_vars(chris.contracted,
                                GRInterval<c_Gam1, c_Gam3>());
	/*current_cell.store_vars(chris.contracted,
                                GRInterval<c_shift1, c_shift3>());*/
    }
};

#endif /* GAMMACALCULATOR_HPP_ */
