/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef MOVINGPUNCTUREGAUGE_HPP_
#define MOVINGPUNCTUREGAUGE_HPP_

#include "DimensionDefinitions.hpp"
#include "Tensor.hpp"
#include "simd.hpp"

/// This is an example of a gauge class that can be used in the GHCRHS compute
/// class
/**
 * This class implements a slightly more generic version of the moving puncture
 * gauge. In particular it uses a Bona-Masso slicing condition of the form
 * f(lapse) = -c*lapse^(p-2)
 * and a Gamma-driver shift condition
 **/
template <class background_t = Minkowski> class MovingPunctureGauge
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
        double shift_Gamma_coeff = 1.; //!< Gives the F in \f$\partial_t
                                       //!  \beta^i =  F B^i\f$
        double shift_advec_coeff = 0.; //!< Switches advection terms in the
                                       //! shift condition on/off
        double eta = 1.; //!< The eta in \f$\partial_t B^i = \partial_t \tilde
                         //!\Gamma - \eta B^i\f$
    };

  protected:
    params_t m_params;
    background_t m_background;

  public:
    MovingPunctureGauge(const params_t &a_params, background_t a_background)
        : m_params(a_params), m_background(a_background)
    {
    }

    template <class data_t, template <typename> class vars_t,
              template <typename> class diff2_vars_t>
    inline void rhs_gauge(vars_t<data_t> &rhs, const vars_t<data_t> &vars,
                          const vars_t<Tensor<1, data_t>> &d1,
                          const diff2_vars_t<Tensor<2, data_t>> &d2,
                          const vars_t<data_t> &advec,
                          const Coordinates<data_t> &coords) const
    {
        using namespace TensorAlgebra;

        Tensor<2, data_t> bg_g;
        Tensor<2, Tensor<1, data_t>> bg_dg;
        m_background.compute_g_and_dg(bg_g, bg_dg, coords);

        auto bg_h_UU = compute_inverse_sym(bg_g);
        auto bg_chris = compute_christoffel(bg_dg, bg_h_UU);

        Tensor<2, data_t> phys_g;
        FOR(i, j) { phys_g[i][j] = vars.h[i][j] + bg_g[i][j]; }

        data_t det_g = compute_determinant_sym(phys_g);
        det_g = sqrt(det_g * det_g);
        data_t chi = pow(det_g, -1. / (double)GR_SPACEDIM);
        data_t chi_regularised = simd_max(1.e-4, chi);

        auto g_UU = compute_inverse_sym(phys_g);
        Tensor<2, Tensor<1, data_t>> d1_phys_g;
	FOR(i, j, k) d1_phys_g[i][j][k] = d1.h[i][j][k] + bg_dg[i][j][k];
	//auto phys_chris = compute_christoffel(d1_phys_g, g_UU);
	
	auto diff_chris = compute_christoffel(d1.h, g_UU);

    	// this will be needed (as all the places where bg_chris appears) until we
    	// use directly the bg covariant derivative of the vars rather than their
    	// partial derivative

    	FOR(i, j, k, l, m)
    	{
            diff_chris.contracted[i] -=
            	bg_chris.ULL[l][j][k] * vars.h[m][l] * g_UU[i][m] * g_UU[j][k];
    	}

	
        rhs.lapse =
            m_params.lapse_advec_coeff * advec.lapse +
            (m_params.lapse_coeff * pow(vars.lapse, m_params.lapse_power)) *
                vars.Pi;
        FOR(i)
        {
            // With conformal gamma
            rhs.shift[i] = m_params.shift_advec_coeff * advec.shift[i] +
                           m_params.shift_Gamma_coeff *
                               diff_chris.contracted[i] / chi_regularised -
                           m_params.eta * vars.shift[i];
            FOR(j, k, l)
            {
                rhs.shift[i] += m_params.shift_Gamma_coeff * g_UU[i][j] *
                                g_UU[k][l] * d1_phys_g[k][l][j] /
                                (6. * chi_regularised);
            }
            rhs.B[i] = 0.;
        }
    }
};

#endif /* MOVINGPUNCTUREGAUGE_HPP_ */
