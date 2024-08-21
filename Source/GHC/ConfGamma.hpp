/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// This compute class calculates Hamiltonian and Momentum constraints

#ifndef CONFGAMMA_HPP_
#define CONFGAMMA_HPP_

#include "GHCVars.hpp"
#include "Cell.hpp"
#include "FArrayBox.H"
#include "FourthOrderDerivatives.hpp"
#include "Tensor.hpp"
#include "simd.hpp"


#include <array>

class ConfGamma
{
  public:
    /// CCZ4 variables
    template <class data_t> using MetricVars = GHCVars::VarsNoGauge<data_t>;

    /// Vars object for Constraints
    template <class data_t> struct Vars
    {
        data_t chi;
	Tensor<1, data_t> CGam;
	Tensor<1, data_t> Z;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
        {
            using namespace VarsTools;
            define_enum_mapping(mapping_function, c_chi, chi);
            define_enum_mapping(mapping_function, GRInterval<c_CGam1, c_CGam3>(),
                                CGam);
	    define_enum_mapping(mapping_function, GRInterval<c_Z1, c_Z3>(),
                                Z);
        }
    };

    ConfGamma(double dx, const Interval &a_c_CGams);

    template <class data_t> void compute(Cell<data_t> current_cell) const;

  protected:
    const FourthOrderDerivatives m_deriv;
    const Interval m_c_CGams;

    template <class data_t, template <typename> class vars_t>
    Vars<data_t>
    conf_gamma_equations(const vars_t<data_t> &vars,
                         const vars_t<Tensor<1, data_t>> &d1,
			 const Tensor<2, data_t> &g_UU,
                         const chris_t<data_t> &chris) const;

    template <class data_t>
    void store_vars(Vars<data_t> &out, Cell<data_t> &current_cell) const;
};

#include "ConfGamma.impl.hpp"

#endif /* CONFGAMMA_HPP_ */
