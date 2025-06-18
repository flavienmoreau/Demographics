# This function generates a dict object, with the keys being 3-digit country ISO codes and values being T * J matrices.
# T: number of periods in the simulation. J: number of age groups in the age-labor productivity profile.
# Each row of the T * J matrix specifies age-specific ratios of labor productivity l_j at the current time period relative to the initial ss.
# Inputs: 
# 1. Convergence regression: TFP growths over time for each country
# 2. Healthy aging: regression coefficients for t, t^2 and t^3
# 3. Spillover from age structure: coefficients for the delta(share 25-45) term
# 4. Female LFP: growth paths of labor supply over time

def gen_h_mult(case = 'Baseline', start_year = 2016, T = 385, beta_conv = -1e10, beta_pop = 0, beta_pop_lag = 0, beta_inter = -1e10, FE_conv = 0, FLFP_gap = None, LIC_wedge=0, migration=False, vintage_UNPP=False, fertility_scenario="medium"):

    from Code.demographics import Population, load_poptrans
    from Code.calibration import load_full_calibration, load_initial_ss
    import pandas as pd, numpy as np
    import copy, math
    
    path_input_data = "@Import/Data/input_data/"
    path_intermediate_data = "@Import/Data/intermediate_data/"
    
    _, gamma, hh_init, prod_init, _, _, _ = load_initial_ss(case="homothetic", LIC_wedge=LIC_wedge)
    h_adj_init = {c: hh_init[c].h_adj for c in hh_init}
    cs = list(h_adj_init.keys())

    #########################################################################################################################    
    # 1. Demographics and innovation
    # log(TFP_{t+1} / TFP_{t}) = beta_pop * log(pop_25_45 / pop_25_45_lag) 
    #########################################################################################################################
    
    h_mult_convergence = {c: np.ones((T, len(h_adj_init[c]))) for c in cs}
    
    if case in ['Baseline', 'Innovation', 'Baseline_morehealth', 'FLFP', 'All', 'No_innovation']:
        
        TFP_dist = pd.read_csv(path_intermediate_data + "TFP.csv").query("year==@start_year")
        TFP_dist_init = TFP_dist[(TFP_dist['year'] == start_year)].set_index('countrycode')[['dist_frontier']]
        TFP_dist_init.loc['LIC','dist_frontier'] = 0.29 
        TFP_dist_init.loc['JPN','dist_frontier'] = 0.71 
    
        pop = load_poptrans(fixed_mortality=False,migration=migration,fertility_scenario=fertility_scenario,vintage_UNPP=vintage_UNPP)
        TFP_dist_path = {c: np.ones((T,)) for c in cs}
        
        suffix = "_mig" if migration else ""
        # Load the population data for 2015 to calculate lagged growth of population 25-45 years old
        df_countries = pd.read_csv(f"@Import/Data/calibration_targets/targets_trans_countries{suffix}.csv")
        df_countries_age = pd.read_csv(f"@Import/Data/calibration_targets/targets_trans_countries_age{suffix}.csv")
        N_2015 = df_countries.query("year == 2015")[['isocode','N']].set_index('isocode')
        pi_2015 = df_countries_age.query("year == 2015 and 25<=age_bin<=45")[['isocode','age_bin','pi']]
        share_25_45_2015 = pi_2015.groupby('isocode')['pi'].sum()
        pop_25_45_2015 = {c: N_2015.loc[c]['N'] * share_25_45_2015.loc[c] for c in N_2015.index}
        
        if case == 'No_innovation':
            beta_pop = 0
            beta_pop_lag = 0
        
        if np.isscalar(FE_conv):
            s = FE_conv
            FE_conv = {c: s for c in cs}

        for c in cs:
            alpha_curr = prod_init[c].alpha
            TFP_dist_path[c][0] = TFP_dist_init.loc[c,'dist_frontier']
            pop_25_45_g_lag = (np.sum(pop[c].pij[0,25:46]) * pop[c].N[0] / pop_25_45_2015[c] - 1) * 100
            
            for t in range(1,T):
                pop_25_45 = np.sum(pop[c].pij[t,25:46]) * pop[c].N[t]
                pop_25_45_lag = np.sum(pop[c].pij[t-1,25:46]) * pop[c].N[t-1]
                pop_25_45_g = (pop_25_45 - pop_25_45_lag) / pop_25_45_lag * 100

                dist_curr = TFP_dist_path[c][t-1]
                
                log_TFP_growth = (beta_conv * math.log(dist_curr) + beta_pop * pop_25_45_g + beta_pop_lag * pop_25_45_g_lag + beta_inter * math.log(dist_curr) * pop_25_45_g + FE_conv[c])
                Z_growth = math.exp(log_TFP_growth) ** (1/(1-alpha_curr))  
                TFP_dist_path[c][t] = dist_curr * math.exp(log_TFP_growth)
                
                # if case == "Demog_only":            # In this option, only return the component of TFP growth related to demographics (TFP dist to frontier still evolves according to the baseline)
                #     Z_growth = math.exp(beta_pop * pop_25_45_g + beta_pop_lag * pop_25_45_g_lag + beta_inter * math.log(dist_curr) * pop_25_45_g) ** (1/(1-alpha_curr)) 
                
                h_mult_convergence[c][t,:] = h_mult_convergence[c][t-1,:] * Z_growth
                pop_25_45_g_lag = pop_25_45_g
    
    #########################################################################################################################
    # 2. Healthy aging
    #########################################################################################################################
    
    h_mult_healthy_aging = {c: np.ones((T, len(h_adj_init[c]))) for c in cs}
    
    if case in ['Baseline', 'Healthy aging', 'Baseline_morehealth', 'FLFP', 'All', 'No_innovation']:
        
        he_suffix = "_alt" if case == 'Baseline_morehealth' or case == 'All' else ""
        healthy_aging = pd.read_csv(path_intermediate_data + f"h_mult_healthy_aging{he_suffix}.csv", header=None)
        
        for c in cs:
            
            for t in range(1,T):

                if t < 85:
                    h_mult_healthy_aging[c][t,50:60] = healthy_aging.iloc[t,0]
                    h_mult_healthy_aging[c][t,60:70] = healthy_aging.iloc[t,1]
                    h_mult_healthy_aging[c][t,70:80] = healthy_aging.iloc[t,2]
                else:
                    h_mult_healthy_aging[c][t,50:60] = h_mult_healthy_aging[c][t-1,50:60]
                    h_mult_healthy_aging[c][t,60:70] = h_mult_healthy_aging[c][t-1,60:70]
                    h_mult_healthy_aging[c][t,70:80] = h_mult_healthy_aging[c][t-1,70:80]
                
    #########################################################################################################################    
    # 3. Increase in female labor force participation
    #########################################################################################################################
    
    h_mult_FLFP = {c: np.ones((T, len(hh_init['USA'].h_adj))) for c in cs}
    
    if case in ['FLFP', 'All']:
        lfpr = pd.read_excel(path_intermediate_data + "scenario_lfpr_long.xlsx", sheet_name = f'{FLFP_gap}_percent').set_index('iso3code')
        year_max, year_min = lfpr['year'].max(), lfpr['year'].min()
        lf_max = lfpr.query('year == @year_max')[['lfpr_15to24', 'lfpr_25to54', 'lfpr_55to64', 'lfpr_65to99']]
        lfpr = lfpr.reset_index().set_index(['iso3code', 'year']) 
        
        for c in cs:
            
            for t in range(1,T):
                year = year_min + t
                if t < year_max - year_min + 1:
                    h_mult_FLFP[c][t,:25] = lfpr.loc[(c,year)]['lfpr_15to24']
                    h_mult_FLFP[c][t,25:55] = lfpr.loc[(c,year)]['lfpr_25to54']
                    h_mult_FLFP[c][t,55:65] = lfpr.loc[(c,year)]['lfpr_55to64']
                    h_mult_FLFP[c][t,65:] = lfpr.loc[(c,year)]['lfpr_65to99']
                else:
                    h_mult_FLFP[c][t,:25] = lf_max.loc[c,'lfpr_15to24']
                    h_mult_FLFP[c][t,25:55] = lf_max.loc[c,'lfpr_25to54']
                    h_mult_FLFP[c][t,55:65] = lf_max.loc[c,'lfpr_55to64']
                    h_mult_FLFP[c][t,65:] = lf_max.loc[c,'lfpr_65to99']
        
            
    ### Final step: multiply all components of growth to get h_mult
    h_mult = {c: h_mult_healthy_aging[c] * h_mult_convergence[c] * h_mult_FLFP[c] for c in cs}
    
    return h_mult