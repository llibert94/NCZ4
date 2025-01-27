
/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef MINKOWSKI_HPP_
#define MINKOWSKI_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "GHCVars.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp" //This files needs NUM_VARS - total number of components
#include "simd.hpp"

//! Class which computes the initial conditions for a Kerr Schild BH
//! https://arxiv.org/pdf/gr-qc/9805023.pdf
//! https://arxiv.org/pdf/2011.07870.pdf

class Minkowski
{
  public:
    template <class data_t> using Vars = GHCVars::VarsWithGauge<data_t>;
    template <class data_t>
    using Diff2Vars = GHCVars::Diff2VarsWithGauge<data_t>;

    Minkowski() {}

    /// This just calculates chi which helps with regridding, debug etc
    /// it is only done once on setup as the BG is fixed
    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        const Coordinates<data_t> coords(current_cell, m_dx, m_params.center);
        Vars<data_t> metric_vars;
        Vars<Tensor<1, data_t>> d1;
        Diff2Vars<Tensor<2, data_t>> d2;
        Tensor<4, data_t> d_chris_ULL;
        Tensor<4, data_t> Riemann;

        compute_metric(metric_vars, d1, d2, d_chris_ULL, Riemann, coords);

        FOR(i) metric_vars.B[i] = 0.;
        // Populate the variables on the grid
        // NB We stil need to set Gamma^i which is NON ZERO
        // but we do this via a separate class/compute function
        // as we need the gradients of the metric which are not yet available
        current_cell.store_vars(metric_vars);
    }

    // Kerr Schild solution
    template <class data_t>
    void compute_g_and_dg(Tensor<2, data_t> &g,
                          Tensor<2, Tensor<1, data_t>> &dg,
                          const Coordinates<data_t> &coords) const
    {
        FOR(i, j) g[i][j] = TensorAlgebra::delta(i, j);
        FOR(i, j, k) dg[i][j][k] = 0.;
    }

    template <class data_t, template <typename> class vars_t,
              template <typename> class diff2_vars_t>
    void compute_metric(vars_t<data_t> &vars, vars_t<Tensor<1, data_t>> &d1,
                        diff2_vars_t<Tensor<2, data_t>> &d2,
                        Tensor<4, data_t> &d_chris_ULL,
                        Tensor<4, data_t> &Riemann,
                        const Coordinates<data_t> &coords) const
    {
        FOR(i, j)
        {
            vars.h[i][j] = TensorAlgebra::delta(i, j);
            vars.K[i][j] = 0.;
            FOR(k)
            {
                d1.h[i][j][k] = 0.;
                d1.K[i][j][k] = 0.;
                FOR(l) d2.h[i][j][k][l] = 0.;
            }
        }
        vars.Pi = 0.;
        vars.lapse = 1.;
        FOR(i)
        {
            d1.lapse[i] = 0.;
            FOR(j) d2.lapse[i][j] = 0.;
        }
        FOR(i)
        {
            vars.shift[i] = 0.;
            FOR(j)
            {
                d1.shift[i][j] = 0.;
                FOR(k) d2.shift[i][j][k] = 0.;
            }
        }
    }
};

#endif /* MINKOWSKI_HPP_ */
