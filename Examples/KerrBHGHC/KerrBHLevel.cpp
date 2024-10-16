/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "KerrBHLevel.hpp"
#include "BoxLoops.hpp"
#include "ConfGamma.hpp"
#include "GHCRHS.hpp"
#include "FixedGridsTaggingCriterion.hpp"
#include "ComputePack.hpp"
#include "KerrBHLevel.hpp"
#include "NanCheck.hpp"
#include "NewConstraints.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "SetValue.hpp"
#include "SixthOrderDerivatives.hpp"
//#include "TraceARemoval.hpp"

// Initial data
#include "GammaCalculator.hpp"
#include "KerrBH.hpp"

#include "ADMQuantities.hpp"
#include "ADMQuantitiesExtraction.hpp"

void KerrBHLevel::specificAdvance()
{
    // Enforce positive chi and alpha
    BoxLoops::loop(PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse),
                   m_state_new, m_state_new, INCLUDE_GHOST_CELLS);
    
    // Check for nan's
    if (m_p.nan_check)
        BoxLoops::loop(
            NanCheck(m_dx, m_p.center, "NaNCheck in specific Advance"),
            m_state_new, m_state_new, EXCLUDE_GHOST_CELLS, disable_simd());
}

void KerrBHLevel::initialData()
{
    CH_TIME("KerrBHLevel::initialData");
    if (m_verbosity)
        pout() << "KerrBHGHCLevel::initialData " << m_level << endl;

    // First set everything to zero then calculate initial data  Get the Kerr
    // solution in the variables, then calculate the \tilde\Gamma^i numerically
    // as these are non zero and not calculated in the Kerr ICs
    BoxLoops::loop(
        make_compute_pack(SetValue(0.), KerrBH(m_p.kerr_params, m_dx)),
        m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    fillAllGhosts();
    BoxLoops::loop(GammaCalculator(m_dx, m_p.center, m_p.kerr_bg), m_state_new, m_state_new,
                   EXCLUDE_GHOST_CELLS);

#ifdef USE_AHFINDER
    // Diagnostics needed for AHFinder
    BoxLoops::loop(Constraints(m_dx, c_Ham, Interval(c_Mom1, c_Mom3)),
                   m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
#endif
}

#ifdef CH_USE_HDF5
void KerrBHLevel::prePlotLevel()
{
#ifdef USE_AHFINDER
    // already calculated in 'specificPostTimeStep'
    if (m_bh_amr.m_ah_finder.need_diagnostics(m_dt, m_time))
        return;
#endif

    fillAllGhosts();
    BoxLoops::loop(ConfGamma(m_dx, Interval(c_CGam1, c_CGam3)),
                    m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
}
#endif /* CH_USE_HDF5 */

void KerrBHLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs,
                                  const double a_time)
{
    // Enforce the trace free A_ij condition and positive chi and alpha
    BoxLoops::loop(PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse),
                   a_soln, a_soln, INCLUDE_GHOST_CELLS);
    
    // Calculate CCZ4 right hand side
    if (m_p.max_spatial_derivative_order == 4)
    {
        BoxLoops::loop(GHCRHS<MovingPunctureGauge, FourthOrderDerivatives>(
                           m_p.ghc_params, m_dx, m_p.sigma, m_p.center, m_p.kerr_bg),
                       a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
    else if (m_p.max_spatial_derivative_order == 6)
    {
        BoxLoops::loop(GHCRHS<MovingPunctureGauge, SixthOrderDerivatives>(
                           m_p.ghc_params, m_dx, m_p.sigma, m_p.center, m_p.kerr_bg),
                       a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
}

void KerrBHLevel::specificUpdateODE(GRLevelData &a_soln,
                                    const GRLevelData &a_rhs, Real a_dt)
{
}

void KerrBHLevel::preTagCells()
{
}

void KerrBHLevel::computeTaggingCriterion(
    FArrayBox &tagging_criterion, const FArrayBox &current_state,
    const FArrayBox &current_state_diagnostics)
{
    BoxLoops::loop(FixedGridsTaggingCriterion(m_dx, m_level, m_p.L, m_p.center),
		   current_state, tagging_criterion);
}

void KerrBHLevel::specificPostTimeStep()
{
    CH_TIME("KerrBHLevel::specificPostTimeStep");
        // Do the extraction on the min extraction level
    if (m_p.activate_extraction == 1)
    {
        int min_level = m_p.extraction_params.min_extraction_level();
        bool calculate_adm = at_level_timestep_multiple(min_level);
        if (calculate_adm)
        {
            // Populate the ADM Mass and Spin values on the grid
            fillAllGhosts();
            BoxLoops::loop(ADMQuantities(m_p.extraction_params.center, m_dx,
                                         c_Madm, c_Jadm),
                           m_state_new, m_state_diagnostics,
                           EXCLUDE_GHOST_CELLS);

            if (m_level == min_level)
            {
                CH_TIME("ADMExtraction");
                // Now refresh the interpolator and do the interpolation
                m_gr_amr.m_interpolator->refresh();
                ADMQuantitiesExtraction my_extraction(
                    m_p.extraction_params, m_dt, m_time, m_restart_time, c_Madm,
                    c_Jadm);
                my_extraction.execute_query(m_gr_amr.m_interpolator);
            }
        }
    }
#ifdef USE_AHFINDER
    // if print is on and there are Diagnostics to write, calculate them!
    if (m_bh_amr.m_ah_finder.need_diagnostics(m_dt, m_time))
    {
        fillAllGhosts();
        BoxLoops::loop(Constraints(m_dx, c_Ham, Interval(c_Mom1, c_Mom3)),
                       m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
    }
    if (m_p.AH_activate && m_level == m_p.AH_params.level_to_run)
        m_bh_amr.m_ah_finder.solve(m_dt, m_time, m_restart_time);
#endif
}
