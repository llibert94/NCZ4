/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GHCRHS_HPP_
#define GHCRHS_HPP_

#include "GHCGeometry.hpp"
#include "GHCVars.hpp"
#include "Cell.hpp"
#include "FourthOrderDerivatives.hpp"
#include "MovingPunctureGauge.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "simd.hpp"

#include "UserVariables.hpp" //This files needs NUM_VARS - total number of components

#include <array>

/// Base parameter struct for CCZ4
/** This struct collects the gauge independent CCZ4 parameters i.e. the damping
 * ones
 */
struct GHC_base_params_t
{
    double kappa1;    //!< Damping parameter kappa1 as in arXiv:1106.2254
    double kappa2;    //!< Damping parameter kappa2 as in arXiv:1106.2254
    double kappa3;    //!< Damping parameter kappa3 as in arXiv:1106.2254
    bool covariantZ4; //!< if true, replace kappa1->kappa1/lapse as in
                      //!<  arXiv:1307.7391 eq. 27
};

/// Parameter struct for CCZ4
/** This struct collects all parameters that are necessary for CCZ4 such as
 * gauge and damping parameters. It inherits from CCZ4_base_params_t and
 * gauge_t::params_t
 */
template <class gauge_params_t = MovingPunctureGauge::params_t>
struct GHC_params_t : public GHC_base_params_t, public gauge_params_t
{
};

/// Compute class to calculate the GHC right hand side
/**
 * This compute class implements the GHC right hand side equations. Use it by
 *handing it to a loop in the BoxLoops namespace. CCZ4RHS includes a struct
 *in its scope: GHCRHS::Vars (the GHC variables).
 **/
template <class gauge_t = MovingPunctureGauge,
          class deriv_t = FourthOrderDerivatives>
class GHCRHS
{
  public:

    using params_t = GHC_params_t<typename gauge_t::params_t>;

    /// GHC variables
    template <class data_t> using Vars = GHCVars::VarsWithGauge<data_t>;

    /// CCZ4 variables
    template <class data_t>
    using Diff2Vars = GHCVars::Diff2VarsWithGauge<data_t>;

  protected:
    const params_t m_params; //!< CCZ4 parameters
    const gauge_t m_gauge;   //!< Class to compute gauge in rhs_equation
    const double m_sigma;    //!< Coefficient for Kreiss-Oliger dissipation
    double m_cosmological_constant;
    const deriv_t m_deriv;

  public:
    /// Constructor
    GHCRHS(
        params_t a_params,            //!< The CCZ4 parameters
        double a_dx,                  //!< The grid spacing
        double a_sigma,               //!< Kreiss-Oliger dissipation coefficient
        double a_cosmological_constant = 0 //!< Value of the cosmological const.
    );

    /// Compute function
    /** This function orchestrates the calculation of the rhs for one specific
     * grid cell. This function is called by the BoxLoops::loop for each grid
     * cell; there should rarely be a need to call it directly.
     */
    template <class data_t> void compute(Cell<data_t> current_cell) const;

  protected:
    /// Calculates the rhs for GHC
    /** Calculates the right hand side for GHC and calls rhs_gauge for the
     *gauge conditions The variables (the template argument vars_t) must contain
     *at least the members: g[i][j], Gam[i], K[i][j], Theta, lapse and
     *shift[i].
     **/
    template <class data_t, template <typename> class vars_t,
              template <typename> class diff2_vars_t>
    void rhs_equation(
        vars_t<data_t> &rhs, //!< Reference to the variables into which the
                             //! output right hand side is written
        const vars_t<data_t> &vars, //!< The values of the current variables
        const vars_t<Tensor<1, data_t>>
            &d1, //!< First derivative of the variables
        const diff2_vars_t<Tensor<2, data_t>>
            &d2, //!< The second derivative the variables
        const vars_t<data_t>
            &advec //!< The advection derivatives of the variables
    ) const;
};

#include "GHCRHS.impl.hpp"
/*
// This is here for backwards compatibility though the CCZ4RHS class should be
// used in future hence mark as deprecated
#ifdef __INTEL_COMPILER
// Intel compiler bug means a spurious warning is printed (tinyurl.com/y9vfgj9j)
// Supress the warning with this pragma
#pragma warning(disable : 2651)
#endif
using CCZ4 [[deprecated("Use CCZ4RHS instead")]] = CCZ4RHS<>;
*/
#endif /* GHCRHS_HPP_ */
