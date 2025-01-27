/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// This compute class calculates some diagnostics

#ifndef CONFORMALDIAGNOSTICS_HPP_
#define CONFORMALDIAGNOSTICS_HPP_

#include "Cell.hpp"
#include "FArrayBox.H"
#include "FourthOrderDerivatives.hpp"
#include "GHCVars.hpp"
#include "Tensor.hpp"
#include "simd.hpp"

#include <array>

template <class background_t> class ConformalDiagnostics
{
  public:
    /// CCZ4 variables
    template <class data_t> using MetricVars = GHCVars::VarsNoGauge<data_t>;

    /// Vars object for Constraints
    template <class data_t> struct Vars
    {
        Tensor<2, data_t> g;
        data_t chi;
        Tensor<1, data_t> CGam;
        Tensor<1, data_t> Z;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
        {
            using namespace VarsTools;
            define_symmetric_enum_mapping(
                mapping_function, GRInterval<c_g11, D_SELECT(, c_g22, c_g33)>(),
                g);
            define_enum_mapping(mapping_function, c_chi, chi);
            define_enum_mapping(mapping_function,
                                GRInterval<c_CGam1, c_CGam3>(), CGam);
            define_enum_mapping(mapping_function, GRInterval<c_Z1, c_Z3>(), Z);
        }
    };

    ConformalDiagnostics(double dx,
                         const std::array<double, CH_SPACEDIM> a_center,
                         background_t a_background, const Interval &a_c_gs,
                         int a_c_chi, const Interval &a_c_CGams = Interval(),
                         const Interval &a_c_Zs = Interval());

    template <class data_t> void compute(Cell<data_t> current_cell) const;

  protected:
    const FourthOrderDerivatives m_deriv;
    const std::array<double, CH_SPACEDIM> m_center;
    const Interval m_c_gs;
    int m_c_chi;
    const Interval m_c_CGams;
    const Interval m_c_Zs;
    background_t m_background;

    template <class data_t, template <typename> class vars_t>
    Vars<data_t>
    conformal_diagnostics_equations(const vars_t<data_t> &vars,
                                    const vars_t<Tensor<1, data_t>> &d1,
                                    const Coordinates<data_t> &coords) const;

    template <class data_t>
    void store_vars(Vars<data_t> &out, Cell<data_t> &current_cell) const;
};

#include "ConformalDiagnostics.impl.hpp"

#endif /* CONFORMALDIAGNOSTICS_HPP_ */
