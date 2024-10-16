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

class PositiveChiAndAlpha
{
  private:
    const double m_min_chi;
    const double m_min_lapse;

  public:
    template <class data_t> struct Vars
    {
        Tensor<2, data_t> g;
        data_t lapse;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
	{
            using namespace VarsTools; // define_enum_mapping is part of
                                       // VarsTools
            define_enum_mapping(mapping_function, c_lapse, lapse);
	    define_symmetric_enum_mapping(
            	mapping_function, GRInterval<c_g11, D_SELECT(, c_g22, c_g33)>(), g);
        }
    };
    //! Constructor for class
    PositiveChiAndAlpha(const double a_min_chi = 1e-4,
                        const double a_min_lapse = 1e-4)
        : m_min_chi(a_min_chi), m_min_lapse(a_min_lapse)
    {
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        auto vars = current_cell.template load_vars<Vars>();
	
	data_t det_g = TensorAlgebra::compute_determinant_sym(vars.g);
	det_g = sqrt(det_g * det_g);
	data_t chi = pow(det_g, -1. / (double)GR_SPACEDIM);

	data_t chi_min = simd_max(chi, m_min_chi);
	FOR(i,j) vars.g[i][j] = vars.g[i][j] * chi / chi_min;
        vars.lapse = simd_max(vars.lapse, m_min_lapse);

	current_cell.store_vars(vars);
    }
};

#endif /* POSITIVECHIANDALPHA_HPP_ */
