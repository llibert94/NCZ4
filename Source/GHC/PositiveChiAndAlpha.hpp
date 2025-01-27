/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// This compute class enforces the positive chi and alpha condition
#ifndef POSITIVECHIANDALPHA_HPP_
#define POSITIVECHIANDALPHA_HPP_

#include "Cell.hpp"
#include "UserVariables.hpp"
#include "simd.hpp"

template <class background_t> class PositiveChiAndAlpha
{
  private:
    double m_dx;
    const std::array<double, CH_SPACEDIM> m_center;
    background_t m_background;
    const double m_min_chi;
    const double m_min_lapse;

  public:
    template <class data_t> struct Vars
    {
        Tensor<2, data_t> h;
	Tensor<2, data_t> K;
        data_t lapse;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
        {
            using namespace VarsTools; // define_enum_mapping is part of
                                       // VarsTools
            define_enum_mapping(mapping_function, c_lapse, lapse);
            define_symmetric_enum_mapping(
                mapping_function, GRInterval<c_h11, D_SELECT(, c_h22, c_h33)>(),
                h);
        }
    };
    //! Constructor for class
    PositiveChiAndAlpha(double a_dx,
                        const std::array<double, CH_SPACEDIM> a_center,
                        background_t a_background,
                        const double a_min_chi = 1e-4,
                        const double a_min_lapse = 1e-4)
        : m_dx(a_dx), m_center(a_center), m_background(a_background),
          m_min_chi(a_min_chi), m_min_lapse(a_min_lapse)
    {
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        auto vars = current_cell.template load_vars<Vars>();

        Tensor<2, data_t> bg_g;
        Tensor<2, Tensor<1, data_t>> bg_dg;
        Coordinates<data_t> coords{current_cell, m_dx, m_center};
        m_background.compute_g_and_dg(bg_g, bg_dg, coords);

        Tensor<2, data_t> phys_g;
        FOR(i, j) phys_g[i][j] = vars.h[i][j] + bg_g[i][j];

        data_t det_g = TensorAlgebra::compute_determinant_sym(phys_g);
        det_g = sqrt(det_g * det_g);
        data_t chi = pow(det_g, -1. / (double)GR_SPACEDIM);

        data_t chi_min = simd_max(chi, m_min_chi);
        FOR(i, j) vars.h[i][j] = phys_g[i][j] * chi / chi_min - bg_g[i][j];
        vars.lapse = simd_max(vars.lapse, m_min_lapse);

        current_cell.store_vars(vars);
    }
};

#endif /* POSITIVECHIANDALPHA_HPP_ */
