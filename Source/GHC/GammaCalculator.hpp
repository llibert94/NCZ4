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

template <class background_t> class GammaCalculator
{
    // Only variables needed are metric
    template <class data_t> struct Vars
    {
        Tensor<2, data_t> h;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
        {
            VarsTools::define_symmetric_enum_mapping(
                mapping_function, GRInterval<c_h11, c_h33>(), h);
        }
    };

  protected:
    const FourthOrderDerivatives
        m_deriv; //!< An object for calculating derivatives of the variables
    const std::array<double, CH_SPACEDIM> m_center;
    background_t m_background;

  public:
    GammaCalculator(double a_dx, const std::array<double, CH_SPACEDIM> a_center,
                    background_t a_background)
        : m_deriv(a_dx), m_center(a_center), m_background(a_background)
    {
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        // copy data from chombo gridpoint into local variables, and calc 1st
        // derivs
        const auto vars = current_cell.template load_vars<Vars>();
        const auto d1 = m_deriv.template diff1<Vars>(current_cell);

        Coordinates<data_t> coords{current_cell, this->m_deriv.m_dx, m_center};

        using namespace TensorAlgebra;

        Tensor<2, data_t> bg_g;
        Tensor<2, Tensor<1, data_t>> bg_dg;
        m_background.compute_g_and_dg(bg_g, bg_dg, coords);

        auto bg_h_UU = compute_inverse_sym(bg_g);
        auto bg_chris = compute_christoffel(bg_dg, bg_h_UU);

        Tensor<2, data_t> phys_g;
        FOR(i, j) { phys_g[i][j] = vars.h[i][j] + bg_g[i][j]; }

        auto g_UU = compute_inverse_sym(phys_g);
        auto diff_chris = compute_christoffel(d1.h, g_UU);
        FOR(i, j, k, l, m)
        diff_chris.contracted[i] -=
            bg_chris.ULL[l][j][k] * vars.h[m][l] * g_UU[i][m] * g_UU[j][k];

        // assign values of Gamma^k = g_UU^ij * \tilde{Gamma}^k_ij in the output
        // FArrayBox
        current_cell.store_vars(diff_chris.contracted,
                                GRInterval<c_Gam1, c_Gam3>());
    }
};

#endif /* GAMMACALCULATOR_HPP_ */
