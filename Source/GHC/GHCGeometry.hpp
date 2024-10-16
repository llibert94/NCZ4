/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// This file calculates GHC geometric quantities (or a similar 3+1 split).
#ifndef GHCGEOMETRY_HPP_
#define GHCGEOMETRY_HPP_

#include "DimensionDefinitions.hpp"
#include "TensorAlgebra.hpp"

//! A structure for the decomposed elements of the Energy Momentum Tensor in
//! 3+1D
template <class data_t> struct emtensor_t
{
    Tensor<2, data_t> Sij; //!< S_ij = T_ij
    Tensor<1, data_t> Si;  //!< S_i = T_ia_n^a
    data_t S;              //!< S = S^i_i
    data_t rho;            //!< rho = T_ab n^a n^b
};

template <class data_t> struct ricci_t
{
    Tensor<2, data_t> LL; // Ricci with two indices down
    data_t scalar;        // Ricci scalar
};

class GHCGeometry
{
  public:
    template <class data_t, template <typename> class vars_t,
              template <typename> class diff2_vars_t>
    static ricci_t<data_t>
    compute_ricci_Z(const vars_t<data_t> &vars,
                    const vars_t<Tensor<1, data_t>> &d1,
                    const diff2_vars_t<Tensor<2, data_t>> &d2,
                    const Tensor<2, data_t> &g_UU, const chris_t<data_t> &chris,
                    const Tensor<1, data_t> &Z,
		    const chris_t<data_t> &bg_chris, const Tensor<4, data_t> &d_bg_chris_ULL,
		    const Tensor<4, data_t> &bg_Riemann,
		    bool kerr_bg)
    {
        ricci_t<data_t> out;

        Tensor<3, data_t> chris_LLU = {0.};
	Tensor<3, data_t> chris_LUU = {0.};

	FOR(i, j, k, l) chris_LLU[i][j][k] += g_UU[k][l] * chris.LLL[i][j][l];
	FOR(i, j, k, l) chris_LUU[i][j][k] += g_UU[j][l] * chris_LLU[i][l][k];

        FOR(i, j)
        {
            out.LL[i][j] = 0.;
            FOR(k)
            {  
		out.LL[i][j] += 0.5 * (vars.g[k][i] * d1.Gam[k][j] +
                                    vars.g[k][j] * d1.Gam[k][i]);
	        out.LL[i][j] += 0.5 * vars.Gam[k] * d1.g[i][j][k];	
                FOR(l)
                {
                    out.LL[i][j] += -0.5 * g_UU[k][l] * d2.g[i][j][k][l] -
                                    chris.LLL[i][k][l] * chris_LUU[j][k][l];
		    FOR(m, n)
		    {
			out.LL[i][j] += g_UU[l][n] * g_UU[k][m] * 
					  d1.g[k][i][l] * d1.g[m][j][n];
		    }
                }
            }
        }
	
	if (kerr_bg)
	{
	    Tensor<3, data_t> bg_chris_LLL = {0.};
	    Tensor<3, data_t> bg_chris_LLU = {0.};
            Tensor<3, data_t> bg_chris_LUU = {0.};

            FOR(i, j, k, l) bg_chris_LLL[i][j][k] += vars.g[i][l] * bg_chris.ULL[l][j][k];
	    FOR(i, j, k, l) bg_chris_LLU[i][j][k] += g_UU[k][l] * bg_chris_LLL[i][j][l];
            FOR(i, j, k, l) bg_chris_LUU[i][j][k] += g_UU[j][l] * bg_chris_LLU[i][l][k];

	    FOR(i, j, k, l)
            {
                out.LL[i][j] += 0.5 * vars.Gam[l] * (vars.g[k][i] * bg_chris.ULL[k][j][l] + 
						     vars.g[k][j] * bg_chris.ULL[k][i][l]);
                out.LL[i][j] -= 0.5 * vars.Gam[k] * (bg_chris.ULL[l][i][k] * vars.g[l][j] + 
						     bg_chris.ULL[l][k][j] * vars.g[i][l]);
		out.LL[i][j] += chris.LLL[i][k][l] * bg_chris_LUU[j][k][l] +
				bg_chris_LLL[i][k][l] * (chris_LUU[j][k][l] -
							 bg_chris_LUU[j][k][l]);
		FOR(m) {
		    out.LL[i][j] += 0.5 * g_UU[k][l] * (bg_chris.ULL[m][k][l] * d1.g[i][j][m] +
						    	bg_chris.ULL[m][k][j] * d1.g[i][m][l] +
						    	bg_chris.ULL[m][k][i] * d1.g[m][j][l]);
                    out.LL[i][j] += 0.5 * g_UU[k][l] * (bg_chris.ULL[m][l][i] * d1.g[m][j][k] + 
				    			d_bg_chris_ULL[m][l][i][k] * vars.g[m][j] +
							bg_chris.ULL[m][j][l] * d1.g[i][m][k] +
							d_bg_chris_ULL[m][j][l][k] * vars.g[i][m]);
		    out.LL[i][j] -= 0.5 * g_UU[k][l] * (vars.g[m][i] * bg_Riemann[m][l][k][j] +
                                                        vars.g[m][j] * bg_Riemann[m][l][k][i]);
		    FOR(n)
		    {
			out.LL[i][j] -= 0.5 * g_UU[k][l] * (bg_chris.ULL[m][n][i] * vars.g[m][j] * bg_chris.ULL[n][k][l] +
							    bg_chris.ULL[m][l][n] * vars.g[m][j] * bg_chris.ULL[n][k][i] +
							    bg_chris.ULL[m][l][i] * vars.g[m][n] * bg_chris.ULL[n][k][j] +
							    bg_chris.ULL[m][j][n] * vars.g[i][m] * bg_chris.ULL[n][k][l] +
                                                            bg_chris.ULL[m][n][l] * vars.g[i][m] * bg_chris.ULL[n][k][j] +
                                                            bg_chris.ULL[m][j][l] * vars.g[n][m] * bg_chris.ULL[n][k][i]);
		    	FOR(p)
                    	{                   
                            out.LL[i][j] -= g_UU[l][n] * g_UU[k][m] *
                            	              (d1.g[k][i][l] * (bg_chris.ULL[p][m][n] * vars.g[p][j] +
							        bg_chris.ULL[p][n][j] * vars.g[m][p]) +
					       d1.g[m][j][n] * (bg_chris.ULL[p][k][l] * vars.g[p][i] +
					   	    		bg_chris.ULL[p][l][i] * vars.g[k][p]));
		            FOR(q)
		            {
			    	out.LL[i][j] += g_UU[l][n] * g_UU[k][m] * 
						(bg_chris.ULL[p][m][n] * vars.g[p][j] +
                                        	 bg_chris.ULL[p][n][j] * vars.g[m][p]) *
						(bg_chris.ULL[q][k][l] * vars.g[q][i] +
                                         	 bg_chris.ULL[q][l][i] * vars.g[k][q]);
		            }
                    	}
		    }
		}
            }
        }

        out.scalar = TensorAlgebra::compute_trace(out.LL, g_UU);

        return out;
    }

    template <class data_t>
    static Tensor<2, data_t>
    compute_d1_chris_contracted(const Tensor<2, data_t> &g_UU,
                                const Tensor<2, Tensor<1, data_t>> &d1_g,
                                const Tensor<2, Tensor<2, data_t>> &d2_g)
    {
        Tensor<2, data_t> d1_chris_contracted = 0.;
        FOR(i, j)
        {
            FOR(m, n, p)
            {
                d1_chris_contracted[i][j] +=
                    g_UU[i][m] * g_UU[n][p] * (d2_g[m][n][j][p] - 0.5 * d2_g[n][p][j][m]);
		FOR(q, r)
		{
		    d1_chris_contracted[i][j] +=
			-d1_g[q][r][j] * (d1_g[m][n][p] - 0.5 * d1_g[n][p][m]) *
				(g_UU[i][m] * g_UU[n][q] * g_UU[p][r] + 
				 g_UU[n][p] * g_UU[i][q] * g_UU[m][r]);

		}
            }
        }
        return d1_chris_contracted;
    }

    // This function allows adding arbitrary multiples of D_{(i}Z_{j)}
    // to the Ricci scalar rather than the default of 2 in compute_ricci_Z
    template <class data_t, template <typename> class vars_t,
              template <typename> class diff2_vars_t>
    static ricci_t<data_t>
    compute_ricci_Z_general(const vars_t<data_t> &vars,
                            const vars_t<Tensor<1, data_t>> &d1,
                            const diff2_vars_t<Tensor<2, data_t>> &d2,
                            const Tensor<2, data_t> &g_UU,
                            const chris_t<data_t> &chris, const double dZ_coeff)
    {
        // get contributions from conformal metric and factor with zero Z vector
        Tensor<1, data_t> Z0 = 0.;
	Tensor<4, data_t> d_chris = {0.};
	Tensor<4, data_t> Riem = {0.};
        auto ricci = compute_ricci_Z(vars, d1, d2, g_UU, chris, Z0, chris, d_chris, Riem, 0);
	// TO BE GENERALISED FOR A BACKGROUND!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        // need to add term to correct for d1.Gamma (includes Z contribution)
        // and Gamma in ricci_hat
        auto d1_chris_contracted =
            compute_d1_chris_contracted(g_UU, d1.g, d2.g);
        Tensor<1, data_t> Z;
        FOR(i) { Z[i] = 0.5 * (vars.Gam[i] - chris.contracted[i]); }
        FOR(i, j)
        {
            FOR(m)
            {
                // This corrects for the \hat{Gamma}s in ricci_hat
                ricci.LL[i][j] +=
                    (1. - 0.5 * dZ_coeff) * 0.5 *
                    (vars.g[m][i] *
                         (d1_chris_contracted[m][j] - d1.Gam[m][j]) +
                     vars.g[m][j] *
                         (d1_chris_contracted[m][i] - d1.Gam[m][i]) +
                     (chris.contracted[m] - vars.Gam[m]) * d1.g[i][j][m]);
            }
        }
        ricci.scalar = TensorAlgebra::compute_trace(ricci.LL, g_UU);
        return ricci;
    }

    // This function returns the pure Ricci scalar with no contribution from the
    // Z vector - used e.g. in the constraint calculations.
    template <class data_t, template <typename> class vars_t,
              template <typename> class diff2_vars_t>
    static ricci_t<data_t>
    compute_ricci(const vars_t<data_t> &vars,
                  const vars_t<Tensor<1, data_t>> &d1,
                  const diff2_vars_t<Tensor<2, data_t>> &d2,
                  const Tensor<2, data_t> &g_UU, const chris_t<data_t> &chris)
    {
        return compute_ricci_Z_general(vars, d1, d2, g_UU, chris, 0.);
    }
};

#endif /* GHCGEOMETRY_HPP_ */
