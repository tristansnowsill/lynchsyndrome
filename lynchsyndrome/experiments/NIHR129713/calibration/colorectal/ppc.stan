functions {
    matrix tp_matrix(
        int       g,
        int       age,
        // MiMiC-Bowel parameters
        vector    norm_lr,
        vector    lr_hr,
        vector    hr_crc,
        real      serrated_max,
        real      rate_presentation_A,
        real      rate_presentation_B,
        real      rate_presentation_C,
        real      rate_presentation_D,
        // Lynch syndrome parameters
        vector    THETA_CONS,
        vector    THETA_AGE,
        vector    ETA,
        vector    PHI,
        vector    PSI,
        real      rho0,
        real      rho1,
        real      rho2,
        real      kappa,
        real      nu,
        real      rate_progression_AB,
        real      rate_progression_BC,
        real      rate_progression_CD
    ) {
        matrix[14,14] tp = rep_matrix(0.0, 14, 14);
        real beta = serrated_max * fmax(age - 15.0, 0) / 85.0;
        real rho = rho0 / (1 + exp(-(age - rho1) / rho2));
        
        tp[1,2] = norm_lr[age];
        if (num_elements(THETA_CONS) >= g) {
            real theta_cons = THETA_CONS[g];
            real theta_age = THETA_AGE[g];
            real cumhaz_theta = exp(theta_cons) / theta_age * (exp(theta_age * (age + 1.0)) - exp(theta_age * age));
            tp[1,4] = 1 - exp(-cumhaz_theta);
        }
        tp[1,7] = beta;
        tp[1,1] = 1.0 - tp[1,2] - tp[1,4] - tp[1,7];

        tp[2,3] = lr_hr[age];
        tp[2,5] = ETA[g];
        tp[2,7] = beta;
        tp[2,2] = 1.0 - tp[2,3] - tp[2,5] - tp[2,7];

        tp[3,6] = ETA[g];
        tp[3,7] = hr_crc[age] + beta;
        tp[3,3] = 1.0 - tp[3,6] - tp[3,7];

        tp[4,1] = rho;
        if (num_elements(PHI) >= g) {
            tp[4,6] = PHI[g];
        }
        if (num_elements(PSI) >= g) {
            tp[4,7] = PSI[g];
        }
        tp[4,7] += beta;
        tp[4,4] = 1.0 - tp[4,1] - tp[4,6] - tp[4,7];

        tp[5,1] = rho;
        tp[5,6] = kappa;
        tp[5,7] = beta;
        tp[5,5] = 1.0 - rho - kappa - beta;

        tp[6,1] = rho;
        tp[6,7] = nu + beta;
        tp[6,6] = 1.0 - rho - nu - beta;

        tp[7,7] = exp(-(rate_progression_AB + rate_presentation_A));
        tp[7,8] = rate_progression_AB / (rate_progression_AB + rate_presentation_A) * (1.0 - tp[7,7]);
        tp[7,11] = rate_presentation_A / (rate_progression_AB + rate_presentation_A) * (1.0 - tp[7,7]);

        tp[8,8] = exp(-(rate_progression_BC + rate_presentation_B));
        tp[8,9] = rate_progression_BC / (rate_progression_BC + rate_presentation_B) * (1.0 - tp[8,8]);
        tp[8,12] = rate_presentation_B / (rate_progression_BC + rate_presentation_B) * (1.0 - tp[8,8]);

        tp[9,9] = exp(-(rate_progression_CD + rate_presentation_C));
        tp[9,10] = rate_progression_CD / (rate_progression_CD + rate_presentation_C) * (1.0 - tp[9,9]);
        tp[9,13] = rate_presentation_C / (rate_progression_CD + rate_presentation_C) * (1.0 - tp[9,9]);

        tp[10,10] = exp(-rate_presentation_D);
        tp[10,14] = 1.0 - tp[10,10];

        tp[11,11] = 1.0;
        tp[12,12] = 1.0;
        tp[13,13] = 1.0;
        tp[14,14] = 1.0;

        if (min(tp) < 0.0)
            reject("Transition probability matrix includes negative element");

        return tp;
    }
}

data {
    // OBSERVATIONS FROM STUDIES

    // Dabir et al. (2020)

    int<lower=0>                    Dabir_N_MLH1;
    int<lower=0>                    Dabir_N_MSH2;
    int<lower=0>                    Dabir_N_MSH6;
    int<lower=0,upper=Dabir_N_MLH1> Dabir_MSI_MLH1;
    int<lower=0,upper=Dabir_N_MSH2> Dabir_MSI_MSH2;
    int<lower=0,upper=Dabir_N_MSH6> Dabir_MSI_MSH6;

    real                            log_risk_ratio_under60;
    real<lower=0>                   se_log_risk_ratio_under60;

    real                            log_risk_ratio_lowrisk;
    real<lower=0>                   se_log_risk_ratio_lowrisk;


    // PLSD (Dominguez-Valentin et al. 2020)
    vector<lower=0>[11]             PLSD_exposure_MLH1_male;
    vector<lower=0>[11]             PLSD_exposure_MSH2_male;
    vector<lower=0>[11]             PLSD_exposure_MSH6_male;
    vector<lower=0>[11]             PLSD_exposure_MLH1_female;
    vector<lower=0>[11]             PLSD_exposure_MSH2_female;
    vector<lower=0>[11]             PLSD_exposure_MSH6_female;
    vector<lower=0>[11]             PLSD_exposure_PMS2;
    array[11] int<lower=0>          PLSD_crc_MLH1_male;
    array[11] int<lower=0>          PLSD_crc_MSH2_male;
    array[11] int<lower=0>          PLSD_crc_MSH6_male;
    array[11] int<lower=0>          PLSD_crc_MLH1_female;
    array[11] int<lower=0>          PLSD_crc_MSH2_female;
    array[11] int<lower=0>          PLSD_crc_MSH6_female;
    array[11] int<lower=0>          PLSD_crc_PMS2;

    // Engel et al. 2018
    int<lower=0>                    Germany_N;
    array[2] int<lower=0>           Germany_sex;
    array[3] int<lower=0>           Germany_genotype;
    real<lower=0>                   Germany_age_index_mean;
    real<lower=0>                   Germany_age_index_sd;
    array[3] int<lower=0>           Germany_index_findings;
    int<lower=0>                    Germany_fu_colonoscopies;
    int<lower=0>                    Germany_fu_adenomas;
    array[4] int<lower=0>           Germany_fu_crc;

    int<lower=0>                    Netherlands_N;
    array[2] int<lower=0>           Netherlands_sex;
    array[3] int<lower=0>           Netherlands_genotype;
    real<lower=0>                   Netherlands_age_index_mean;
    real<lower=0>                   Netherlands_age_index_sd;
    array[3] int<lower=0>           Netherlands_index_findings;
    int<lower=0>                    Netherlands_fu_colonoscopies;
    int<lower=0>                    Netherlands_fu_adenomas;
    array[4] int<lower=0>           Netherlands_fu_crc;

    int<lower=0>                    Finland_N;
    array[2] int<lower=0>           Finland_sex;
    array[3] int<lower=0>           Finland_genotype;
    real<lower=0>                   Finland_age_index_mean;
    real<lower=0>                   Finland_age_index_sd;
    array[3] int<lower=0>           Finland_index_findings;
    int<lower=0>                    Finland_fu_colonoscopies;
    int<lower=0>                    Finland_fu_adenomas;
    array[4] int<lower=0>           Finland_fu_crc;


    // PARAMETERS FROM MIMIC-BOWEL TREATED AS CONSTANTS
    array[2] vector[100]            norm_lr;
    array[2] vector[100]            lr_hr;
    array[2] vector[100]            hr_crc;
    array[2] real                   serrated_max;
    real                            progression_AB;
    real                            progression_BC;
    real                            progression_CD;
    real                            presentation_A;
    real                            presentation_B;
    real                            presentation_C;
    real                            presentation_D;
    real                            col_sens_lr;
    real                            col_sens_hr;
    real                            col_sens_crc;
}

transformed data {
    // MIMIC-BOWEL PARAMETERS
    real total_hazard_DukesA = -log1m(progression_AB + presentation_A);
    real total_hazard_DukesB = -log1m(progression_BC + presentation_B);
    real total_hazard_DukesC = -log1m(progression_CD + presentation_C);
    real total_hazard_DukesD = -log1m(presentation_D);

    real rate_presentation_A = presentation_A / (progression_AB + presentation_A) * total_hazard_DukesA;
    real rate_presentation_B = presentation_B / (progression_BC + presentation_B) * total_hazard_DukesB;
    real rate_presentation_C = presentation_C / (progression_CD + presentation_C) * total_hazard_DukesC;
    real rate_presentation_D = total_hazard_DukesD;

    // Engel et al. (2018)
    // Gauss-Hermite regression
    //   We have premultiplied x by sqrt(2) and predivided w by sqrt(pi)
    //   so that if X ~ N(mu, sigma) then
    //   E[f(X)] \approx \sum_{i=1}^{3}{w_i f(mu + sigma * x_i)}
    vector[3] gh_x = [-1.224745,0,1.224745]';
    vector[3] gh_w = [1.0/6.0,6.0/9.0,1.0/6.0]';

    vector[3] Germany_age_index = Germany_age_index_mean + Germany_age_index_sd * gh_x;
    vector[3] Netherlands_age_index = Netherlands_age_index_mean + Netherlands_age_index_sd * gh_x;
    vector[3] Finland_age_index = Finland_age_index_mean + Finland_age_index_sd * gh_x;
}

parameters {
    // Adenoma
    vector<upper=-1>[3]             theta_cons;
    vector<lower=0,upper=0.05>[3]   theta_age;
    real<lower=0,upper=0.5>         rho0;
    real                            rho1;
    real<lower=0>                   rho2;
    vector<lower=0,upper=0.5>[4]    eta;
    vector<lower=0,upper=0.5>[3]    phi;
    vector<lower=0,upper=0.5>[3]    psi;
    real<lower=0,upper=0.5>         kappa;
    real<lower=0,upper=0.5>         nu;

    // CRC
    real<lower=0>                   rate_LS_progression_AB;
    real<lower=0>                   rate_LS_progression_BC;
    real<lower=0>                   rate_LS_progression_CD;

    // Relative proportions of path_MLH1, path_MSH2 and path_MSH6 in
    // Dabir et al. (2020)
    vector[3]                       Dabir_genotype;
}

generated quantities {
    real p_MSI_MLH1;
    real p_MSI_MSH2;
    real p_MSI_MSH6;
    real pred_log_risk_ratio_under60;
    real pred_log_risk_ratio_lowrisk;

    int ppc_Dabir_MSI_MLH1;
    int ppc_Dabir_MSI_MSH2;
    int ppc_Dabir_MSI_MSH6;
    real ppc_risk_ratio_under60;
    real ppc_risk_ratio_lowrisk;

    array[11] int ppc_PLSD_crc_MLH1_male;
    array[11] int ppc_PLSD_crc_MLH1_female;
    array[11] int ppc_PLSD_crc_MSH2_male;
    array[11] int ppc_PLSD_crc_MSH2_female;
    array[11] int ppc_PLSD_crc_MSH6_male;
    array[11] int ppc_PLSD_crc_MSH6_female;
    array[11] int ppc_PLSD_crc_PMS2;

    array[3]  int ppc_Germany_index_findings;
    array[3]  int ppc_Netherlands_index_findings;
    array[3]  int ppc_Finland_index_findings;

    int ppc_Germany_fu_adenomas;
    int ppc_Netherlands_fu_adenomas;
    int ppc_Finland_fu_adenomas;

    array[4]  int ppc_Germany_fu_crc;
    array[4]  int ppc_Netherlands_fu_crc;
    array[4]  int ppc_Finland_fu_crc;

    {
        // ============================
        // SIMULATIONS
        // ============================

        // Dabir et al. (2020)
        int age_start_lower = 25;
        int age_start_step = 50;
        int n_age_start = 6;

        array[3,2] real n_MSI_under60  = rep_array(0.0, 3, 2);
        array[3,2] real n_MSS_under60  = rep_array(0.0, 3, 2);
        array[3,2] real n_MSI_over60   = rep_array(0.0, 3, 2);
        array[3,2] real n_MSS_over60   = rep_array(0.0, 3, 2);
        array[3,2] real n_MSI_lowrisk  = rep_array(0.0, 3, 2);
        array[3,2] real n_MSS_lowrisk  = rep_array(0.0, 3, 2);
        array[3,2] real n_MSI_highrisk = rep_array(0.0, 3, 2);
        array[3,2] real n_MSS_highrisk = rep_array(0.0, 3, 2);

        for (g in 1:3) {
            for (s in 1:2) {
                for (i in 1:n_age_start) {
                    int age_start_colonoscopy = age_start_lower + (i - 1) * age_start_step;
                    row_vector[14] state = one_hot_row_vector(14, 1);
                    for (age in 1:70) {
                        // Update state
                        matrix[14,14] tp = tp_matrix(
                            g, age,
                            // MiMiC-Bowel parameters
                            norm_lr[s], lr_hr[s], hr_crc[s], serrated_max[s],
                            rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                            // Lynch syndrome parameters
                            theta_cons, theta_age, eta, phi, psi,
                            rho0, rho1, rho2, kappa, nu,
                            rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                        );
                        state = state * tp;
                        // Apply colonoscopy if relevant
                        if ((age >= age_start_colonoscopy) && ((age - age_start_colonoscopy) % 2 == 0)) {
                            // Adenomas
                            real detected_lowrisk_MSS = col_sens_lr * state[2];
                            real detected_lowrisk_MSI = col_sens_lr * state[5];
                            real detected_highrisk_MSS = col_sens_hr * state[3];
                            real detected_highrisk_MSI = col_sens_hr * state[6];

                            state[1] += detected_lowrisk_MSS + detected_lowrisk_MSI +
                                detected_highrisk_MSS + detected_highrisk_MSI;
                            state[2] -= detected_lowrisk_MSS;
                            state[3] -= detected_highrisk_MSS;
                            state[5] -= detected_lowrisk_MSI;
                            state[6] -= detected_highrisk_MSI;

                            n_MSS_lowrisk[g,s]  += detected_lowrisk_MSS;
                            n_MSS_highrisk[g,s] += detected_highrisk_MSS;
                            n_MSI_lowrisk[g,s]  += detected_lowrisk_MSI;
                            n_MSI_highrisk[g,s] += detected_highrisk_MSI;

                            if (age < 60) {
                                n_MSS_under60[g,s] += detected_lowrisk_MSS + detected_highrisk_MSS;
                                n_MSI_under60[g,s] += detected_lowrisk_MSI + detected_highrisk_MSI;
                            } else {
                                n_MSS_over60[g,s] += detected_lowrisk_MSS + detected_highrisk_MSI;
                                n_MSI_over60[g,s] += detected_lowrisk_MSI + detected_highrisk_MSI;
                            }
                        }
                    }
                }
            }
        }

        real odds_MSI_MLH1 = (n_MSI_lowrisk[1,1] + n_MSI_lowrisk[1,2] + n_MSI_highrisk[1,1] + n_MSI_highrisk[1,2]) /
            (n_MSS_lowrisk[1,1] + n_MSS_lowrisk[1,2] + n_MSS_highrisk[1,1] + n_MSS_highrisk[1,2]);
        real odds_MSI_MSH2 = (n_MSI_lowrisk[2,1] + n_MSI_lowrisk[2,2] + n_MSI_highrisk[2,1] + n_MSI_highrisk[2,2]) /
            (n_MSS_lowrisk[2,1] + n_MSS_lowrisk[2,2] + n_MSS_highrisk[2,1] + n_MSS_highrisk[2,2]);
        real odds_MSI_MSH6 = (n_MSI_lowrisk[3,1] + n_MSI_lowrisk[3,2] + n_MSI_highrisk[3,1] + n_MSI_highrisk[3,2]) /
            (n_MSS_lowrisk[3,1] + n_MSS_lowrisk[3,2] + n_MSS_highrisk[3,1] + n_MSS_highrisk[3,2]);
        
        real odds_under60 = ((to_row_vector(n_MSI_under60[,1]) + to_row_vector(n_MSI_under60[,2])) * Dabir_genotype) /
            ((to_row_vector(n_MSS_under60[,1]) + to_row_vector(n_MSS_under60[,2])) * Dabir_genotype);
        real odds_over60 = ((to_row_vector(n_MSI_over60[,1]) + to_row_vector(n_MSI_over60[,2])) * Dabir_genotype) /
            ((to_row_vector(n_MSS_over60[,1]) + to_row_vector(n_MSS_over60[,2])) * Dabir_genotype);
        pred_log_risk_ratio_under60 = log(odds_under60) + log1p(odds_over60) - log1p(odds_under60) - log(odds_over60);

        real odds_lowrisk = ((to_row_vector(n_MSI_lowrisk[,1]) + to_row_vector(n_MSI_lowrisk[,2])) * Dabir_genotype) /
            ((to_row_vector(n_MSS_lowrisk[,1]) + to_row_vector(n_MSS_lowrisk[,2])) * Dabir_genotype);
        real odds_highrisk = ((to_row_vector(n_MSI_highrisk[,1]) + to_row_vector(n_MSI_highrisk[,2])) * Dabir_genotype) /
            ((to_row_vector(n_MSS_highrisk[,1]) + to_row_vector(n_MSS_highrisk[,2])) * Dabir_genotype);
        pred_log_risk_ratio_lowrisk = log(odds_lowrisk) + log1p(odds_highrisk) - log1p(odds_lowrisk) - log(odds_highrisk);

        p_MSI_MLH1 = odds_MSI_MLH1 / (1 + odds_MSI_MLH1);
        p_MSI_MSH2 = odds_MSI_MSH2 / (1 + odds_MSI_MSH2);
        p_MSI_MSH6 = odds_MSI_MSH6 / (1 + odds_MSI_MSH6);

        // PLSD (Dominguez-Valentin et al. 2020)

        // MLH1 - males
        vector[11] S_PLSD_MLH1_male;
        vector[11] p_PLSD_MLH1_male = rep_vector(0.0, 11);
        {
            array[11] int age_lower = {25,30,35,40,45,50,55,60,65,70,75};
            array[11] int age_upper = {29,34,39,44,49,54,59,64,69,74,79};

            row_vector[14] state = one_hot_row_vector(14, 1);
            for (age in 1:24) {
                // State evolves with no intervention
                matrix[14,14] tp = tp_matrix(
                    1, age,
                    // MiMiC-Bowel parameters
                    norm_lr[1], lr_hr[1], hr_crc[1], serrated_max[1],
                    rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                    // Lynch syndrome parameters
                    theta_cons, theta_age, eta, phi, psi,
                    rho0, rho1, rho2, kappa, nu,
                    rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                );
                state = state * tp;        
            }
            for (i in 1:11) {
                // Zero any already diagnosed CRC
                state[11:14] = rep_row_vector(0.0, 4);
                // Store probability of having no diagnosed cancer by start
                // of age range
                S_PLSD_MLH1_male[i] = sum(state[1:10]);
                for (age in age_lower[i]:age_upper[i]) {
                    // Colonoscopy at start of every odd year
                    if (age % 2 == 1) {
                        // Adenomas silently moved back to healthy epithelium state
                        state[1] += col_sens_lr * (state[2] + state[5]);
                        state[1] += col_sens_hr * (state[3] + state[6]);
                        state[2] *= 1.0 - col_sens_lr;
                        state[3] *= 1.0 - col_sens_hr;
                        state[5] *= 1.0 - col_sens_lr;
                        state[6] *= 1.0 - col_sens_hr;

                        // Colorectal cancers moved to diagnosed CRC state
                        state[11] += col_sens_crc * state[7];
                        state[12] += col_sens_crc * state[8];
                        state[13] += col_sens_crc * state[9];
                        state[14] += col_sens_crc * state[10];
                        state[7]  *= 1.0 - col_sens_crc;
                        state[8]  *= 1.0 - col_sens_crc;
                        state[9]  *= 1.0 - col_sens_crc;
                        state[10] *= 1.0 - col_sens_crc;
                    }
                    // State evolution
                    matrix[14,14] tp = tp_matrix(
                        1, age,
                        // MiMiC-Bowel parameters
                        norm_lr[1], lr_hr[1], hr_crc[1], serrated_max[1],
                        rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                        // Lynch syndrome parameters
                        theta_cons, theta_age, eta, phi, psi,
                        rho0, rho1, rho2, kappa, nu,
                        rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                    );
                    state = state * tp;
                }
                // Sum all CRC diagnosed in the age range
                p_PLSD_MLH1_male[i] = sum(state[11:14]);
            }
        }
        // MLH1 - females
        vector[11] S_PLSD_MLH1_female;
        vector[11] p_PLSD_MLH1_female = rep_vector(0.0, 11);
        {
            array[11] int age_lower = {25,30,35,40,45,50,55,60,65,70,75};
            array[11] int age_upper = {29,34,39,44,49,54,59,64,69,74,79};

            row_vector[14] state = one_hot_row_vector(14, 1);
            for (age in 1:24) {
                // State evolves with no intervention
                matrix[14,14] tp = tp_matrix(
                    1, age,
                    // MiMiC-Bowel parameters
                    norm_lr[2], lr_hr[2], hr_crc[2], serrated_max[2],
                    rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                    // Lynch syndrome parameters
                    theta_cons, theta_age, eta, phi, psi,
                    rho0, rho1, rho2, kappa, nu,
                    rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                );
                state = state * tp;        
            }
            for (i in 1:11) {
                // Zero any already diagnosed CRC
                state[11:14] = rep_row_vector(0.0, 4);
                // Store probability of having no diagnosed cancer by start
                // of age range
                S_PLSD_MLH1_female[i] = sum(state[1:10]);
                for (age in age_lower[i]:age_upper[i]) {
                    // Colonoscopy at start of every odd year
                    if (age % 2 == 1) {
                        // Adenomas silently moved back to healthy epithelium state
                        state[1] += col_sens_lr * (state[2] + state[5]);
                        state[1] += col_sens_hr * (state[3] + state[6]);
                        state[2] *= 1.0 - col_sens_lr;
                        state[3] *= 1.0 - col_sens_hr;
                        state[5] *= 1.0 - col_sens_lr;
                        state[6] *= 1.0 - col_sens_hr;

                        // Colorectal cancers moved to diagnosed CRC state
                        state[11] += col_sens_crc * state[7];
                        state[12] += col_sens_crc * state[8];
                        state[13] += col_sens_crc * state[9];
                        state[14] += col_sens_crc * state[10];
                        state[7]  *= 1.0 - col_sens_crc;
                        state[8]  *= 1.0 - col_sens_crc;
                        state[9]  *= 1.0 - col_sens_crc;
                        state[10] *= 1.0 - col_sens_crc;
                    }
                    // State evolution
                    matrix[14,14] tp = tp_matrix(
                        1, age,
                        // MiMiC-Bowel parameters
                        norm_lr[2], lr_hr[2], hr_crc[2], serrated_max[2],
                        rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                        // Lynch syndrome parameters
                        theta_cons, theta_age, eta, phi, psi,
                        rho0, rho1, rho2, kappa, nu,
                        rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                    );
                    state = state * tp;
                }
                // Sum all CRC diagnosed in the age range
                p_PLSD_MLH1_female[i] = sum(state[11:14]);
            }
        }

        // MSH2 - males
        vector[11] S_PLSD_MSH2_male;
        vector[11] p_PLSD_MSH2_male = rep_vector(0.0, 11);
        {
            array[11] int age_lower = {25,30,35,40,45,50,55,60,65,70,75};
            array[11] int age_upper = {29,34,39,44,49,54,59,64,69,74,79};

            row_vector[14] state = one_hot_row_vector(14, 1);
            for (age in 1:24) {
                // State evolves with no intervention
                matrix[14,14] tp = tp_matrix(
                    2, age,
                    // MiMiC-Bowel parameters
                    norm_lr[1], lr_hr[1], hr_crc[1], serrated_max[1],
                    rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                    // Lynch syndrome parameters
                    theta_cons, theta_age, eta, phi, psi,
                    rho0, rho1, rho2, kappa, nu,
                    rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                );
                state = state * tp;        
            }
            for (i in 1:11) {
                // Zero any already diagnosed CRC
                state[11:14] = rep_row_vector(0.0, 4);
                // Store probability of having no diagnosed cancer by start
                // of age range
                S_PLSD_MSH2_male[i] = sum(state[1:10]);
                for (age in age_lower[i]:age_upper[i]) {
                    // Colonoscopy at start of every odd year
                    if (age % 2 == 1) {
                        // Adenomas silently moved back to healthy epithelium state
                        state[1] += col_sens_lr * (state[2] + state[5]);
                        state[1] += col_sens_hr * (state[3] + state[6]);
                        state[2] *= 1.0 - col_sens_lr;
                        state[3] *= 1.0 - col_sens_hr;
                        state[5] *= 1.0 - col_sens_lr;
                        state[6] *= 1.0 - col_sens_hr;

                        // Colorectal cancers moved to diagnosed CRC state
                        state[11] += col_sens_crc * state[7];
                        state[12] += col_sens_crc * state[8];
                        state[13] += col_sens_crc * state[9];
                        state[14] += col_sens_crc * state[10];
                        state[7]  *= 1.0 - col_sens_crc;
                        state[8]  *= 1.0 - col_sens_crc;
                        state[9]  *= 1.0 - col_sens_crc;
                        state[10] *= 1.0 - col_sens_crc;
                    }
                    // State evolution
                    matrix[14,14] tp = tp_matrix(
                        2, age,
                        // MiMiC-Bowel parameters
                        norm_lr[1], lr_hr[1], hr_crc[1], serrated_max[1],
                        rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                        // Lynch syndrome parameters
                        theta_cons, theta_age, eta, phi, psi,
                        rho0, rho1, rho2, kappa, nu,
                        rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                    );
                    state = state * tp;
                }
                // Sum all CRC diagnosed in the age range
                p_PLSD_MSH2_male[i] = sum(state[11:14]);
            }
        }
        // MSH2 - females
        vector[11] S_PLSD_MSH2_female;
        vector[11] p_PLSD_MSH2_female = rep_vector(0.0, 11);
        {
            array[11] int age_lower = {25,30,35,40,45,50,55,60,65,70,75};
            array[11] int age_upper = {29,34,39,44,49,54,59,64,69,74,79};

            row_vector[14] state = one_hot_row_vector(14, 1);
            for (age in 1:24) {
                // State evolves with no intervention
                matrix[14,14] tp = tp_matrix(
                    2, age,
                    // MiMiC-Bowel parameters
                    norm_lr[2], lr_hr[2], hr_crc[2], serrated_max[2],
                    rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                    // Lynch syndrome parameters
                    theta_cons, theta_age, eta, phi, psi,
                    rho0, rho1, rho2, kappa, nu,
                    rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                );
                state = state * tp;        
            }
            for (i in 1:11) {
                // Zero any already diagnosed CRC
                state[11:14] = rep_row_vector(0.0, 4);
                // Store probability of having no diagnosed cancer by start
                // of age range
                S_PLSD_MSH2_female[i] = sum(state[1:10]);
                for (age in age_lower[i]:age_upper[i]) {
                    // Colonoscopy at start of every odd year
                    if (age % 2 == 1) {
                        // Adenomas silently moved back to healthy epithelium state
                        state[1] += col_sens_lr * (state[2] + state[5]);
                        state[1] += col_sens_hr * (state[3] + state[6]);
                        state[2] *= 1.0 - col_sens_lr;
                        state[3] *= 1.0 - col_sens_hr;
                        state[5] *= 1.0 - col_sens_lr;
                        state[6] *= 1.0 - col_sens_hr;

                        // Colorectal cancers moved to diagnosed CRC state
                        state[11] += col_sens_crc * state[7];
                        state[12] += col_sens_crc * state[8];
                        state[13] += col_sens_crc * state[9];
                        state[14] += col_sens_crc * state[10];
                        state[7]  *= 1.0 - col_sens_crc;
                        state[8]  *= 1.0 - col_sens_crc;
                        state[9]  *= 1.0 - col_sens_crc;
                        state[10] *= 1.0 - col_sens_crc;
                    }
                    // State evolution
                    matrix[14,14] tp = tp_matrix(
                        2, age,
                        // MiMiC-Bowel parameters
                        norm_lr[2], lr_hr[2], hr_crc[2], serrated_max[2],
                        rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                        // Lynch syndrome parameters
                        theta_cons, theta_age, eta, phi, psi,
                        rho0, rho1, rho2, kappa, nu,
                        rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                    );
                    state = state * tp;
                }
                // Sum all CRC diagnosed in the age range
                p_PLSD_MSH2_female[i] = sum(state[11:14]);
            }
        }

        // MSH6 - males
        vector[11] S_PLSD_MSH6_male;
        vector[11] p_PLSD_MSH6_male = rep_vector(0.0, 11);
        {
            array[11] int age_lower = {25,30,35,40,45,50,55,60,65,70,75};
            array[11] int age_upper = {29,34,39,44,49,54,59,64,69,74,79};

            row_vector[14] state = one_hot_row_vector(14, 1);
            for (age in 1:24) {
                // State evolves with no intervention
                matrix[14,14] tp = tp_matrix(
                    1, age,
                    // MiMiC-Bowel parameters
                    norm_lr[1], lr_hr[1], hr_crc[1], serrated_max[1],
                    rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                    // Lynch syndrome parameters
                    theta_cons, theta_age, eta, phi, psi,
                    rho0, rho1, rho2, kappa, nu,
                    rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                );
                state = state * tp;        
            }
            for (i in 1:11) {
                // Zero any already diagnosed CRC
                state[11:14] = rep_row_vector(0.0, 4);
                // Store probability of having no diagnosed cancer by start
                // of age range
                S_PLSD_MSH6_male[i] = sum(state[1:10]);
                for (age in age_lower[i]:age_upper[i]) {
                    // Colonoscopy at start of every odd year
                    if (age % 2 == 1) {
                        // Adenomas silently moved back to healthy epithelium state
                        state[1] += col_sens_lr * (state[2] + state[5]);
                        state[1] += col_sens_hr * (state[3] + state[6]);
                        state[2] *= 1.0 - col_sens_lr;
                        state[3] *= 1.0 - col_sens_hr;
                        state[5] *= 1.0 - col_sens_lr;
                        state[6] *= 1.0 - col_sens_hr;

                        // Colorectal cancers moved to diagnosed CRC state
                        state[11] += col_sens_crc * state[7];
                        state[12] += col_sens_crc * state[8];
                        state[13] += col_sens_crc * state[9];
                        state[14] += col_sens_crc * state[10];
                        state[7]  *= 1.0 - col_sens_crc;
                        state[8]  *= 1.0 - col_sens_crc;
                        state[9]  *= 1.0 - col_sens_crc;
                        state[10] *= 1.0 - col_sens_crc;
                    }
                    // State evolution
                    matrix[14,14] tp = tp_matrix(
                        1, age,
                        // MiMiC-Bowel parameters
                        norm_lr[1], lr_hr[1], hr_crc[1], serrated_max[1],
                        rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                        // Lynch syndrome parameters
                        theta_cons, theta_age, eta, phi, psi,
                        rho0, rho1, rho2, kappa, nu,
                        rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                    );
                    state = state * tp;
                }
                // Sum all CRC diagnosed in the age range
                p_PLSD_MSH6_male[i] = sum(state[11:14]);
            }
        }
        // MSH6 - females
        vector[11] S_PLSD_MSH6_female;
        vector[11] p_PLSD_MSH6_female = rep_vector(0.0, 11);
        {
            array[11] int age_lower = {25,30,35,40,45,50,55,60,65,70,75};
            array[11] int age_upper = {29,34,39,44,49,54,59,64,69,74,79};

            row_vector[14] state = one_hot_row_vector(14, 1);
            for (age in 1:24) {
                // State evolves with no intervention
                matrix[14,14] tp = tp_matrix(
                    3, age,
                    // MiMiC-Bowel parameters
                    norm_lr[2], lr_hr[2], hr_crc[2], serrated_max[2],
                    rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                    // Lynch syndrome parameters
                    theta_cons, theta_age, eta, phi, psi,
                    rho0, rho1, rho2, kappa, nu,
                    rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                );
                state = state * tp;        
            }
            for (i in 1:11) {
                // Zero any already diagnosed CRC
                state[11:14] = rep_row_vector(0.0, 4);
                // Store probability of having no diagnosed cancer by start
                // of age range
                S_PLSD_MSH6_female[i] = sum(state[1:10]);
                for (age in age_lower[i]:age_upper[i]) {
                    // Colonoscopy at start of every odd year
                    if (age % 2 == 1) {
                        // Adenomas silently moved back to healthy epithelium state
                        state[1] += col_sens_lr * (state[2] + state[5]);
                        state[1] += col_sens_hr * (state[3] + state[6]);
                        state[2] *= 1.0 - col_sens_lr;
                        state[3] *= 1.0 - col_sens_hr;
                        state[5] *= 1.0 - col_sens_lr;
                        state[6] *= 1.0 - col_sens_hr;

                        // Colorectal cancers moved to diagnosed CRC state
                        state[11] += col_sens_crc * state[7];
                        state[12] += col_sens_crc * state[8];
                        state[13] += col_sens_crc * state[9];
                        state[14] += col_sens_crc * state[10];
                        state[7]  *= 1.0 - col_sens_crc;
                        state[8]  *= 1.0 - col_sens_crc;
                        state[9]  *= 1.0 - col_sens_crc;
                        state[10] *= 1.0 - col_sens_crc;
                    }
                    // State evolution
                    matrix[14,14] tp = tp_matrix(
                        3, age,
                        // MiMiC-Bowel parameters
                        norm_lr[2], lr_hr[2], hr_crc[2], serrated_max[2],
                        rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                        // Lynch syndrome parameters
                        theta_cons, theta_age, eta, phi, psi,
                        rho0, rho1, rho2, kappa, nu,
                        rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                    );
                    state = state * tp;
                }
                // Sum all CRC diagnosed in the age range
                p_PLSD_MSH6_female[i] = sum(state[11:14]);
            }
        }

        // PMS2 - persons
        vector[11] S_PLSD_PMS2;
        vector[11] p_PLSD_PMS2 = rep_vector(0.0, 11);
        {
            array[11] int age_lower = {25,30,35,40,45,50,55,60,65,70,75};
            array[11] int age_upper = {29,34,39,44,49,54,59,64,69,74,79};

            row_vector[14] state = one_hot_row_vector(14, 1);
            for (age in 1:24) {
                // State evolves with no intervention
                matrix[14,14] tp = tp_matrix(
                    4, age,
                    // MiMiC-Bowel parameters
                    0.5 * (norm_lr[1] + norm_lr[2]),
                    0.5 * (lr_hr[1] + lr_hr[2]),
                    0.5 * (hr_crc[1] + hr_crc[2]),
                    0.5 * (serrated_max[1] + serrated_max[2]),
                    rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                    // Lynch syndrome parameters
                    theta_cons, theta_age, eta, phi, psi,
                    rho0, rho1, rho2, kappa, nu,
                    rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                );
                state = state * tp;        
            }
            for (i in 1:11) {
                // Zero any already diagnosed CRC
                state[11:14] = rep_row_vector(0.0, 4);
                // Store probability of having no diagnosed cancer by start
                // of age range
                S_PLSD_PMS2[i] = sum(state[1:10]);
                for (age in age_lower[i]:age_upper[i]) {
                    // Colonoscopy at start of every odd year
                    if (age % 2 == 1) {
                        // Adenomas silently moved back to healthy epithelium state
                        state[1] += col_sens_lr * (state[2] + state[5]);
                        state[1] += col_sens_hr * (state[3] + state[6]);
                        state[2] *= 1.0 - col_sens_lr;
                        state[3] *= 1.0 - col_sens_hr;
                        state[5] *= 1.0 - col_sens_lr;
                        state[6] *= 1.0 - col_sens_hr;

                        // Colorectal cancers moved to diagnosed CRC state
                        state[11] += col_sens_crc * state[7];
                        state[12] += col_sens_crc * state[8];
                        state[13] += col_sens_crc * state[9];
                        state[14] += col_sens_crc * state[10];
                        state[7]  *= 1.0 - col_sens_crc;
                        state[8]  *= 1.0 - col_sens_crc;
                        state[9]  *= 1.0 - col_sens_crc;
                        state[10] *= 1.0 - col_sens_crc;
                    }
                    // State evolution
                    matrix[14,14] tp = tp_matrix(
                        4, age,
                        // MiMiC-Bowel parameters
                        0.5 * (norm_lr[1] + norm_lr[2]),
                        0.5 * (lr_hr[1] + lr_hr[2]),
                        0.5 * (hr_crc[1] + hr_crc[2]),
                        0.5 * (serrated_max[1] + serrated_max[2]),
                        rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                        // Lynch syndrome parameters
                        theta_cons, theta_age, eta, phi, psi,
                        rho0, rho1, rho2, kappa, nu,
                        rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                    );
                    state = state * tp;
                }
                // Sum all CRC diagnosed in the age range
                p_PLSD_PMS2[i] = sum(state[11:14]);
            }
        }

        // Engel et al. (2018)
        vector[3] p_Germany_index = zeros_vector(3);
        real      exp_Germany_fu_adenoma = 0;
        vector[4] p_Germany_crc = zeros_vector(4);

        for (g in 1:3) {
            for (s in 1:2) {
                for (i in 1:3) {
                    row_vector[14] state = one_hot_row_vector(14,1);
                    real age_index_colonoscopy = Germany_age_index[i];
                    real n_fu_colonoscopies = 0.0;
                    real n_fu_adenomas = 0.0;
                    real N = gh_w[i] * Germany_sex[s] * Germany_genotype[g] / (Germany_N ^ 2);
                    // print("g=", g);
                    // print("s=", s);
                    // print("i=", i);
                    // print("N=", N);
                    // print("age_index_colonoscopy=", age_index_colonoscopy);
                    for (age in 1:70) {
                        if (age + 1 < age_index_colonoscopy) {
                            // Prior to index colonoscopy
                            // .. Nothing special to do
                        } else if (age > age_index_colonoscopy) {
                            // Follow-up after index colonoscopy
                            // There is a colonoscopy every year...
                            // 1. Count colonoscopy
                            real q_miss = 1929.0 / 2534.0;
                            // 1. Count colonoscopy
                            n_fu_colonoscopies += q_miss * sum(state[1:10]);
                            // 2. Count adenomas
                            real detected_adenoma = q_miss * (col_sens_lr * (state[2] + state[5]) +
                                col_sens_hr * (state[3] + state[6]));
                            n_fu_adenomas += detected_adenoma;
                            state[1]  += detected_adenoma;
                            state[2]  *= q_miss * col_sens_lr;
                            state[3]  *= q_miss * col_sens_hr;
                            state[5]  *= q_miss * col_sens_lr;
                            state[6]  *= q_miss * col_sens_hr;
                            // 3. Detect CRC (they will be counted later)
                            state[11] += col_sens_crc * state[7];
                            state[12] += col_sens_crc * state[8];
                            state[13] += col_sens_crc * state[9];
                            state[14] += col_sens_crc * state[10];
                            state[7]  *= 1.0 - col_sens_crc;
                            state[8]  *= 1.0 - col_sens_crc;
                            state[9]  *= 1.0 - col_sens_crc;
                            state[10] *= 1.0 - col_sens_crc;
                        } else {
                            // Index colonoscopy
                            // 1. Remove any with diagnosed CRC
                            state[11:14] = zeros_row_vector(4);
                            // 2. Reweight remaining population
                            state *= N / sum(state);
                            // 3. Extract relevant probabilities for index findings
                            real detected_adenoma = col_sens_lr * (state[2] + state[5]) +
                                col_sens_hr * (state[3] + state[6]);
                            state[1] += detected_adenoma;
                            real detected_crc = col_sens_crc * sum(state[7:10]);
                            p_Germany_index[1] += N - detected_adenoma - detected_crc;
                            p_Germany_index[2] += detected_adenoma;
                            p_Germany_index[3] += detected_crc;
                            state[2] *= 1.0 - col_sens_lr;
                            state[3] *= 1.0 - col_sens_hr;
                            state[5] *= 1.0 - col_sens_lr;
                            state[6] *= 1.0 - col_sens_hr;
                            state[11] += col_sens_crc * state[7];
                            state[12] += col_sens_crc * state[8];
                            state[13] += col_sens_crc * state[9];
                            state[14] += col_sens_crc * state[10];
                            state[7]  *= 1.0 - col_sens_crc;
                            state[8]  *= 1.0 - col_sens_crc;
                            state[9]  *= 1.0 - col_sens_crc;
                            state[10] *= 1.0 - col_sens_crc;
                        }
                        // Evolve state
                        matrix[14,14] tp = tp_matrix(
                            g, age,
                            // MiMiC-Bowel parameters
                            norm_lr[s], lr_hr[s], hr_crc[s], serrated_max[s],
                            rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                            // Lynch syndrome parameters
                            theta_cons, theta_age, eta, phi, psi,
                            rho0, rho1, rho2, kappa, nu,
                            rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                        );
                        state = state * tp;
                    }
                    exp_Germany_fu_adenoma += N * n_fu_adenomas / n_fu_colonoscopies;
                    p_Germany_crc[1] += state[11];
                    p_Germany_crc[2] += state[12];
                    p_Germany_crc[3] += state[13];
                    p_Germany_crc[4] += state[14];

                    // print("p_Germany_index=", p_Germany_index);
                    // print("p_Germany_crc=", p_Germany_crc);
                    // print("exp_Germany_fu_adenoma=", exp_Germany_fu_adenoma);
                }
            }
        }

        vector[3] p_Netherlands_index = zeros_vector(3);
        real      exp_Netherlands_fu_adenoma = 0;
        vector[4] p_Netherlands_crc = zeros_vector(4);

        for (g in 1:3) {
            for (s in 1:2) {
                for (i in 1:3) {
                    row_vector[14] state = one_hot_row_vector(14,1);
                    real age_index_colonoscopy = Netherlands_age_index[i];
                    real n_fu_colonoscopies = 0.0;
                    real n_fu_adenomas = 0.0;
                    real N = gh_w[i] * Netherlands_sex[s] * Netherlands_genotype[g] / (Netherlands_N ^ 2);
                    for (age in 1:70) {
                        if (age + 1 < age_index_colonoscopy) {
                            // Prior to index colonoscopy
                            // .. Nothing special to do
                        } else if (age > age_index_colonoscopy) {
                            // Follow-up after index colonoscopy
                            // There is a colonoscopy every other year...
                            if (age % 2 == 0) {
                                // 1. Count colonoscopy
                                n_fu_colonoscopies += sum(state[1:10]);
                                // 2. Count adenomas
                                real detected_adenoma = col_sens_lr * (state[2] + state[5]) +
                                    col_sens_hr * (state[3] + state[6]);
                                n_fu_adenomas += detected_adenoma;
                                state[1]  += detected_adenoma;
                                state[2]  *= col_sens_lr;
                                state[3]  *= col_sens_hr;
                                state[5]  *= col_sens_lr;
                                state[6]  *= col_sens_hr;
                                // 3. Detect CRC (they will be counted later)
                                state[11] += col_sens_crc * state[7];
                                state[12] += col_sens_crc * state[8];
                                state[13] += col_sens_crc * state[9];
                                state[14] += col_sens_crc * state[10];
                                state[7]  *= 1.0 - col_sens_crc;
                                state[8]  *= 1.0 - col_sens_crc;
                                state[9]  *= 1.0 - col_sens_crc;
                                state[10] *= 1.0 - col_sens_crc;
                            }
                        } else {
                            // Index colonoscopy
                            // 1. Remove any with diagnosed CRC
                            state[11:14] = zeros_row_vector(4);
                            // 2. Reweight remaining population
                            state *= N / sum(state);
                            // 3. Extract relevant probabilities for index findings
                            real detected_adenoma = col_sens_lr * (state[2] + state[5]) +
                                col_sens_hr * (state[3] + state[6]);
                            state[1] += detected_adenoma;
                            real detected_crc = col_sens_crc * sum(state[7:10]);
                            p_Netherlands_index[1] += N - detected_adenoma - detected_crc;
                            p_Netherlands_index[2] += detected_adenoma;
                            p_Netherlands_index[3] += detected_crc;
                            state[2] *= 1.0 - col_sens_lr;
                            state[3] *= 1.0 - col_sens_hr;
                            state[5] *= 1.0 - col_sens_lr;
                            state[6] *= 1.0 - col_sens_hr;
                            state[11] += col_sens_crc * state[7];
                            state[12] += col_sens_crc * state[8];
                            state[13] += col_sens_crc * state[9];
                            state[14] += col_sens_crc * state[10];
                            state[7]  *= 1.0 - col_sens_crc;
                            state[8]  *= 1.0 - col_sens_crc;
                            state[9]  *= 1.0 - col_sens_crc;
                            state[10] *= 1.0 - col_sens_crc;
                        }
                        // Evolve state
                        matrix[14,14] tp = tp_matrix(
                            g, age,
                            // MiMiC-Bowel parameters
                            norm_lr[s], lr_hr[s], hr_crc[s], serrated_max[s],
                            rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                            // Lynch syndrome parameters
                            theta_cons, theta_age, eta, phi, psi,
                            rho0, rho1, rho2, kappa, nu,
                            rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                        );
                        state = state * tp;
                    }
                    exp_Netherlands_fu_adenoma += N * n_fu_adenomas / n_fu_colonoscopies;
                    p_Netherlands_crc[1] += state[11];
                    p_Netherlands_crc[2] += state[12];
                    p_Netherlands_crc[3] += state[13];
                    p_Netherlands_crc[4] += state[14];
                }
            }
        }

        vector[3] p_Finland_index = zeros_vector(3);
        real      exp_Finland_fu_adenoma = 0;
        vector[4] p_Finland_crc = zeros_vector(4);

        for (g in 1:3) {
            for (s in 1:2) {
                for (i in 1:3) {
                    row_vector[14] state = one_hot_row_vector(14,1);
                    real age_index_colonoscopy = Finland_age_index[i];
                    real n_fu_colonoscopies = 0.0;
                    real n_fu_adenomas = 0.0;
                    real N = gh_w[i] * Finland_sex[s] * Finland_genotype[g] / (Finland_N ^ 2);
                    // print("=======");
                    // print("g=", g);
                    // print("s=", s);
                    // print("i=", i);
                    // print("gh_w[i]=", gh_w[i]);
                    // print("Finland_N=", Finland_N);
                    // print("Finland_genotype[g]=", Finland_genotype[g]);
                    // print("Finland_sex[s]=", Finland_sex[s]);
                    // print("N=", N);
                    for (age in 1:70) {
                        if (age + 1 <= age_index_colonoscopy) {
                            // Prior to index colonoscopy
                            // .. Nothing special to do
                        } else if (age > age_index_colonoscopy) {
                            // Follow-up after index colonoscopy
                            // There are roughly 2 colonoscopies every 5 years...
                            if (age % 5 == 0 || age % 5 == 2) {
                                // 1. Count colonoscopy
                                n_fu_colonoscopies += sum(state[1:10]);
                                // 2. Count adenomas
                                real detected_adenoma = col_sens_lr * (state[2] + state[5]) +
                                    col_sens_hr * (state[3] + state[6]);
                                n_fu_adenomas += detected_adenoma;
                                state[1]  += detected_adenoma;
                                state[2]  *= col_sens_lr;
                                state[3]  *= col_sens_hr;
                                state[5]  *= col_sens_lr;
                                state[6]  *= col_sens_hr;
                                // 3. Detect CRC (they will be counted later)
                                state[11] += col_sens_crc * state[7];
                                state[12] += col_sens_crc * state[8];
                                state[13] += col_sens_crc * state[9];
                                state[14] += col_sens_crc * state[10];
                                state[7]  *= 1.0 - col_sens_crc;
                                state[8]  *= 1.0 - col_sens_crc;
                                state[9]  *= 1.0 - col_sens_crc;
                                state[10] *= 1.0 - col_sens_crc;
                            }
                        } else {
                            // Index colonoscopy
                            // 1. Remove any with diagnosed CRC
                            state[11:14] = zeros_row_vector(4);
                            // 2. Reweight remaining population
                            state *= N / sum(state);
                            // 3. Extract relevant probabilities for index findings
                            real detected_adenoma = col_sens_lr * (state[2] + state[5]) +
                                col_sens_hr * (state[3] + state[6]);
                            state[1] += detected_adenoma;
                            real detected_crc = col_sens_crc * sum(state[7:10]);
                            p_Finland_index[1] += N - detected_adenoma - detected_crc;
                            p_Finland_index[2] += detected_adenoma;
                            p_Finland_index[3] += detected_crc;
                            state[2] *= 1.0 - col_sens_lr;
                            state[3] *= 1.0 - col_sens_hr;
                            state[5] *= 1.0 - col_sens_lr;
                            state[6] *= 1.0 - col_sens_hr;
                            state[11] += col_sens_crc * state[7];
                            state[12] += col_sens_crc * state[8];
                            state[13] += col_sens_crc * state[9];
                            state[14] += col_sens_crc * state[10];
                            state[7]  *= 1.0 - col_sens_crc;
                            state[8]  *= 1.0 - col_sens_crc;
                            state[9]  *= 1.0 - col_sens_crc;
                            state[10] *= 1.0 - col_sens_crc;
                        }
                        // Evolve state
                        matrix[14,14] tp = tp_matrix(
                            g, age,
                            // MiMiC-Bowel parameters
                            norm_lr[s], lr_hr[s], hr_crc[s], serrated_max[s],
                            rate_presentation_A, rate_presentation_B, rate_presentation_C, rate_presentation_D,
                            // Lynch syndrome parameters
                            theta_cons, theta_age, eta, phi, psi,
                            rho0, rho1, rho2, kappa, nu,
                            rate_LS_progression_AB, rate_LS_progression_BC, rate_LS_progression_CD
                        );
                        state = state * tp;
                    }
                    exp_Finland_fu_adenoma += N * n_fu_adenomas / n_fu_colonoscopies;
                    p_Finland_crc[1] += state[11];
                    p_Finland_crc[2] += state[12];
                    p_Finland_crc[3] += state[13];
                    p_Finland_crc[4] += state[14];
                    // print("p_Finland_index", p_Finland_index);
                }
            }
        }

        // =============================
        // SIMULATE POSTERIOR PREDICTIVE
        // =============================

        // Dabir et al. (2020)
        ppc_Dabir_MSI_MLH1 = binomial_rng(Dabir_N_MLH1, p_MSI_MLH1);
        ppc_Dabir_MSI_MSH2 = binomial_rng(Dabir_N_MSH2, p_MSI_MSH2);
        ppc_Dabir_MSI_MSH6 = binomial_rng(Dabir_N_MSH6, p_MSI_MSH6);
        ppc_risk_ratio_under60 = lognormal_rng(pred_log_risk_ratio_under60, se_log_risk_ratio_under60);
        ppc_risk_ratio_lowrisk = lognormal_rng(pred_log_risk_ratio_lowrisk, se_log_risk_ratio_lowrisk);

        // PLSD (Dominguez-Valentin et al. 2020)
        ppc_PLSD_crc_MLH1_male = poisson_rng(-log1m(p_PLSD_MLH1_male ./ S_PLSD_MLH1_male) .* PLSD_exposure_MLH1_male / 5.0);
        ppc_PLSD_crc_MLH1_female = poisson_rng(-log1m(p_PLSD_MLH1_female ./ S_PLSD_MLH1_female) .* PLSD_exposure_MLH1_female / 5.0);
        ppc_PLSD_crc_MSH2_male = poisson_rng(-log1m(p_PLSD_MSH2_male ./ S_PLSD_MSH2_male) .* PLSD_exposure_MSH2_male / 5.0);
        ppc_PLSD_crc_MSH2_female = poisson_rng(-log1m(p_PLSD_MSH2_female ./ S_PLSD_MSH2_female) .* PLSD_exposure_MSH2_female / 5.0);
        ppc_PLSD_crc_MSH6_male = poisson_rng(-log1m(p_PLSD_MSH6_male ./ S_PLSD_MSH6_male) .* PLSD_exposure_MSH6_male / 5.0);
        ppc_PLSD_crc_MSH6_female = poisson_rng(-log1m(p_PLSD_MSH6_female ./ S_PLSD_MSH6_female) .* PLSD_exposure_MSH6_female / 5.0);
        ppc_PLSD_crc_PMS2 = poisson_rng(-log1m(p_PLSD_PMS2 ./ S_PLSD_PMS2) .* PLSD_exposure_PMS2 / 5.0);
        
        // Engel et al. (2018)
        ppc_Germany_index_findings = multinomial_rng(p_Germany_index, sum(Germany_index_findings));
        ppc_Netherlands_index_findings = multinomial_rng(p_Netherlands_index, sum(Netherlands_index_findings));
        ppc_Finland_index_findings = multinomial_rng(p_Finland_index, sum(Finland_index_findings));

        ppc_Germany_fu_adenomas = poisson_rng(Germany_fu_colonoscopies * exp_Germany_fu_adenoma);
        ppc_Netherlands_fu_adenomas = poisson_rng(Netherlands_fu_colonoscopies * exp_Netherlands_fu_adenoma);
        ppc_Finland_fu_adenomas = poisson_rng(Finland_fu_colonoscopies * exp_Finland_fu_adenoma);

        ppc_Germany_fu_crc = multinomial_rng(p_Germany_crc / sum(p_Germany_crc), sum(Germany_fu_crc));
        ppc_Netherlands_fu_crc = multinomial_rng(p_Netherlands_crc / sum(p_Netherlands_crc), sum(Netherlands_fu_crc));
        ppc_Finland_fu_crc = multinomial_rng(p_Finland_crc / sum(p_Finland_crc), sum(Finland_fu_crc));
    }
}
