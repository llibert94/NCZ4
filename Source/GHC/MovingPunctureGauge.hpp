/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef MOVINGPUNCTUREGAUGE_HPP_
#define MOVINGPUNCTUREGAUGE_HPP_

#include "DimensionDefinitions.hpp"
#include "simd.hpp"
#include "Tensor.hpp"

/// This is an example of a gauge class that can be used in the GHCRHS compute
/// class
/**
 * This class implements a slightly more generic version of the moving puncture
 * gauge. In particular it uses a Bona-Masso slicing condition of the form
 * f(lapse) = -c*lapse^(p-2)
 * and a Gamma-driver shift condition
 **/
class MovingPunctureGauge
{
  public:
    struct params_t
    {
        // lapse params:
        double lapse_advec_coeff = 0.; //!< Switches advection terms in
                                       //! the lapse condition on/off
        double lapse_power = 1.; //!< The power p in \f$\partial_t \alpha = - c
                                 //!\alpha^p(K-2\Theta)\f$
        double lapse_coeff = 2.; //!< The coefficient c in \f$\partial_t \alpha
                                 //!= -c \alpha^p(K-2\Theta)\f$
        // shift params:
        double shift_Gamma_coeff = 1.;   //!< Gives the F in \f$\partial_t
                                         //!  \beta^i =  F B^i\f$
        double shift_advec_coeff = 0.;   //!< Switches advection terms in the
                                         //! shift condition on/off
        double eta = 1.; //!< The eta in \f$\partial_t B^i = \partial_t \tilde
                         //!\Gamma - \eta B^i\f$
        double lapidusPower = 1.;
        double lapidusCoeff = 0.1;
        double diffCutoff = 0.03;
        double diffCFLFact = 1e20;

    };

  protected:
    params_t m_params;

  public:
    MovingPunctureGauge(const params_t &a_params) : m_params(a_params) {}

    template <class data_t, template <typename> class vars_t,
              template <typename> class diff2_vars_t>
    inline void rhs_gauge(vars_t<data_t> &rhs, const vars_t<data_t> &vars,
                          const vars_t<Tensor<1, data_t>> &d1,
                          const diff2_vars_t<Tensor<2, data_t>> &d2,
                          const vars_t<data_t> &advec) const
    {
        using namespace TensorAlgebra;

	auto g_UU = compute_inverse_sym(vars.g);
	auto chris = compute_christoffel(d1.g, g_UU);
	data_t tr_K = compute_trace(vars.K, g_UU);

	data_t det_g = compute_determinant_sym(vars.g);
	det_g = sqrt(det_g * det_g);
        data_t chi = pow(det_g, -1. / (double)GR_SPACEDIM);
	data_t chi_regularised = simd_max(1.e-4, chi);

	rhs.lapse = m_params.lapse_advec_coeff * advec.lapse +
                    (m_params.lapse_coeff *
                        pow(vars.lapse, m_params.lapse_power)) * vars.Pi;
        FOR(i)
        {
	    // Not integrated
            /*rhs.shift[i] = m_params.shift_advec_coeff * advec.shift[i] +
                           m_params.shift_Gamma_coeff * vars.B[i];
            rhs.B[i] = m_params.shift_advec_coeff * advec.B[i] +
                       rhs.Gam[i] - m_params.shift_advec_coeff * advec.Gam[i] -
                       m_params.eta * vars.B[i];
	    */
	    // With conformal gamma
	    rhs.shift[i] = m_params.shift_advec_coeff * advec.shift[i] +
                       m_params.shift_Gamma_coeff * chris.contracted[i] / chi_regularised -
                       m_params.eta * vars.shift[i];
            FOR(j,k,l) {
            	rhs.shift[i] += m_params.shift_Gamma_coeff * g_UU[i][j] * g_UU[k][l] * d1.g[k][l][j] / (6. * chi_regularised); 
            }

	    // Integrated
	    /*rhs.shift[i] = m_params.shift_advec_coeff * advec.shift[i] +
                       m_params.shift_Gamma_coeff * vars.Gam[i] -
                       m_params.eta * vars.shift[i];*/
	    rhs.B[i] = 0.;
        }
    }
};

#endif /* MOVINGPUNCTUREGAUGE_HPP_ */
