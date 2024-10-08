/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef ADMQUANTITIES_HPP_
#define ADMQUANTITIES_HPP_

#include "GHCVars.hpp"
#include "GHCGeometry.hpp"
#include "Cell.hpp"
#include "Coordinates.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GRInterval.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp" //This files needs NUM_VARS - total number of components
#include "VarsTools.hpp"
#include "simd.hpp"

//! Calculates the ADM mass
class ADMQuantities
{
    // Use the variable definition in GHCVars - only require the key
    // vars
    template <class data_t> using Vars = GHCVars::VarsNoGauge<data_t>;

    template <class data_t>
    using Diff1Vars = GHCVars::Diff2VarsNoGauge<data_t>;

  public:
    enum DIR
    {
        X,
        Y,
        Z
    };

    ADMQuantities(const std::array<double, CH_SPACEDIM> &a_center, double a_dx,
                  int a_c_Madm = -1, int a_c_Jadm = -1, double a_G_Newton = 1.0)
        : m_deriv(a_dx), m_center(a_center), m_G_Newton(a_G_Newton),
          m_c_Madm(a_c_Madm), m_c_Jadm(a_c_Jadm), m_dir(Z)
    {
    }

    // in case user wants to change direction of spin calculation to something
    // other than Z
    void set_spin_dir(DIR spin_direction) { m_dir = spin_direction; }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        // copy data from chombo gridpoint into local variables, and calc 1st
        // derivs
        const auto vars = current_cell.template load_vars<Vars>();
        const auto d1 = m_deriv.template diff1<Diff1Vars>(current_cell);

        using namespace TensorAlgebra;
        const auto g_UU = compute_inverse_sym(vars.g);

        // Surface element for integration
        Coordinates<data_t> coords(current_cell, m_deriv.m_dx, m_center);
        Tensor<1, data_t> x = {coords.x, coords.y, coords.z};
        Tensor<1, data_t> dS_U = x;

        data_t dS_norm = 0.;
        FOR(i, j) { dS_norm += vars.g[i][j] * dS_U[i] * dS_U[j]; }
        dS_norm = sqrt(dS_norm);
        FOR(i) { dS_U[i] /= dS_norm; }

        Tensor<1, data_t> dS_L;
        FOR(i)
        {
            // dS_L[i] = dS_U[i];
            dS_L[i] = 0.;
            FOR(j) { dS_L[i] += vars.g[i][j] * dS_U[j]; }
        }

        if (m_c_Madm >= 0)
        {
            data_t Madm = 0.0;
            FOR(i, j, k, l)
            {
                Madm += dS_L[i] / (16. * M_PI * m_G_Newton) *
                        g_UU[j][k] * g_UU[i][l] *
                        (d1.g[l][k][j] - d1.g[j][k][l]);
            }

            // assign values of ADM Mass in output box
            current_cell.store_vars(Madm, m_c_Madm);
        }

        if (m_c_Jadm >= 0)
        {
            // spin about m_dir axis (x, y or z)
            data_t Jadm = 0.0;

            // note this is the levi civita symbol,
            // not tensor (eps_tensor = eps_symbol * chi^-1.5)
            const Tensor<3, double> epsilon = TensorAlgebra::epsilon();

	    data_t trK = TensorAlgebra::compute_trace(vars.K, g_UU);

            FOR(i, j, k)
            {
                Jadm += -dS_L[i] / (8. * M_PI * m_G_Newton) *
                        epsilon[m_dir][j][k] * x[j] * trK *
                        TensorAlgebra::delta(i, k);

                FOR(l, m)
                {
                    Jadm += dS_L[i] / (8. * M_PI * m_G_Newton) *
                            epsilon[m_dir][j][k] * x[j] * g_UU[i][l] *
                            g_UU[k][m] * vars.K[l][m];
                }
            }

            // assign values of ADM Momentum in output box
            current_cell.store_vars(Jadm, m_c_Jadm);
        }
    }

  protected:
    const FourthOrderDerivatives
        m_deriv; //!< An object for calculating derivatives of the variables
    const std::array<double, CH_SPACEDIM> &m_center;
    const double m_G_Newton; //!< Newton's constant
    const int m_c_Madm, m_c_Jadm;

    DIR m_dir;
};

#endif /* ADMQUANTITIES_HPP_ */
