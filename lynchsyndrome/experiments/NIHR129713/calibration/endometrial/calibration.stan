functions {
  // Calculate a convolution of sequences x and y
  //
  // Requires that x and y are the same length (M)
  //
  // z[i] = sum(x[j] * y[i-j+1] for j in 1 to i)
  vector convolve(vector x, vector y) {
    int M = num_elements(x);
    vector[M] yrev = reverse(y);
    // We reverse y for convenience
    vector[M] z;
    for (i in 1:M) {
      z[i] = dot_product(x[1:i], yrev[(M-i+1):]);
    }
    return z;
  }
}

data {
    // Age and spline basis
    int<lower=2>            N_age;      // Number of age points
    int<lower=1>            N_basis;    // Number of spline bases
    positive_ordered[N_age] age;        // Age at age points (these should be equally spaced)
    matrix[N_age,N_basis]   age_spline; // The age splines

    // PLSD (Dominguez-Valentin et al. 2020) data
    int<lower=1>                     N_age_PLSD;
    int<lower=1>                     N_gt_PLSD;
    array[N_age_PLSD] int            age_index_lower_PLSD;
    array[N_age_PLSD] int            age_index_upper_PLSD;
    array[N_gt_PLSD,N_age_PLSD] real patient_years_PLSD;
    array[N_gt_PLSD,N_age_PLSD] int  events_PLSD;

    // Cross-sectional data
    array[5] int stage_dist_RenkonenSinisalo2007;
    array[5] int stage_dist_Eikenboom2021;

    // Cancer Registration Statistics, England (2017)
    int<lower=1> N_CRS;
    array[N_CRS] int  age_index_lower_CRS;
    array[N_CRS] int  age_index_upper_CRS;
    array[N_CRS] real patient_years_CRS;
    array[N_CRS] int  events_CRS;

    // English endometrial cancer stage data
    array[4] int stage_dist_EC_England;
}

transformed data {
    real age_step = age[2] - age[1];
}

parameters {
    // Model for AEH incidence
    // We have five different sets of alpha:
    //   alpha[1] = path_MLH1
    //   alpha[2] = path_MSH2
    //   alpha[3] = path_MSH6
    //   alpha[4] = path_PMS2
    //   alpha[5] = No Lynch syndrome
    array[5] vector[N_basis] alpha;

    // Model for AEH development
    real<lower=0>       rho;
    positive_ordered[5] xi;
    vector<lower=0>[4]  lambda;
    vector<lower=0>[5]  delta;
}

transformed parameters {
    // Incidence intensity function
    array[5] vector[N_age] h_AEH_incidence;
    for (i in 1:5)
        h_AEH_incidence[i] = exp(age_spline * alpha[i]);

    // Time-to-diagnosis probability density function
    vector[N_age] f_diagnosed_EC;
    f_diagnosed_EC[1] = 0.0;

    {
        // AEH development ODE matrix
        matrix[16,16] A = rep_matrix(0.0, 16, 16);
        array[N_age] vector[16] y;
        y[1] = one_hot_vector(16, 1);

        A[1,1] = -(rho + xi[1] + lambda[1]);
        A[2,1] = xi[1];
        A[2,2] = -(rho + delta[1] + lambda[1]);
        A[3,2] = delta[1];
        A[4,1] = rho;
        A[4,2] = rho;

        A[5,1] = lambda[1];

        A[5,5] = -(xi[2] + lambda[2]);
        A[6,5] = lambda[2];
        A[6,6] = -(xi[3] + lambda[3]);
        A[7,6] = lambda[3];
        A[7,7] = -(xi[4] + lambda[4]);
        A[8,7] = lambda[4];
        A[8,8] = -xi[5];

        A[9,2] = lambda[1];

        A[9,5] = xi[2];
        A[10,6] = xi[3];
        A[11,7] = xi[4];
        A[12,8] = xi[5];

        A[9,9] = -(delta[2] + lambda[2]);
        A[10,9] = lambda[2];
        A[10,10] = -(delta[3] + lambda[3]);
        A[11,10] = lambda[3];
        A[11,11] = -(delta[4] + lambda[4]);
        A[12,11] = lambda[4];
        A[12,12] = -delta[5];

        A[13,9] = delta[2];
        A[14,10] = delta[3];
        A[15,11] = delta[4];
        A[16,12] = delta[5];

        matrix[16,16] expA = matrix_exp(A);

        matrix[16,1] y0 = to_matrix(one_hot_vector(16, 1));
        row_vector[16] mask = rep_row_vector(0.0, 16);
        mask[13:16] = [1,1,1,1];
        row_vector[16] mask_x_A = mask * A;
        for (i in 2:N_age) {
            y[i] = expA * y[i-1];
            f_diagnosed_EC[i] = mask_x_A * y[i];
        }
    }

    // Endometrial cancer diagnosis intensity function
    // !!!! TODO !!!!
    // -- This will need scaling generally. It seems to work when the age
    //    step is 1, but it will fail if not adjusted.
    array[5] vector[N_age] h_EC_diagnosis;
    for (i in 1:5)
        h_EC_diagnosis[i] = convolve(h_AEH_incidence[i], f_diagnosed_EC);

    // Endometrial cancer diagnosis mean value function and
    // cumulative distribution function
    array[5] vector[N_age] H_EC_diagnosis;
    array[5] vector[N_age] F_EC_diagnosis;
    for (i in 1:5) {
        H_EC_diagnosis[i,1] = 0.0;
        for (j in 2:N_age) {
            H_EC_diagnosis[i,j] = H_EC_diagnosis[i,j-1] +
                age_step * 0.5 * (h_EC_diagnosis[i,j-1] + h_EC_diagnosis[i,j]);
        }
        F_EC_diagnosis[i] = 1.0 - exp(-H_EC_diagnosis[i]);
    }

    // Expected events in PLSD
    array[N_gt_PLSD,N_age_PLSD] real expected_events_PLSD;
    for (g in 1:N_gt_PLSD) {
        for (i in 1:N_age_PLSD) {
            int  i_l = age_index_lower_PLSD[i];
            int  i_u = age_index_upper_PLSD[i];
            real t_l = age[i_l];
            real t_u = age[i_u];
            real patients = patient_years_PLSD[g,i] / (t_u - t_l);
            expected_events_PLSD[g,i] = patients * (H_EC_diagnosis[g,i_u] - H_EC_diagnosis[g,i_l]);
        }
    }

    // Expected events in Cancer Registration Statistics
    array[N_CRS] real expected_events_CRS;
    for (i in 1:N_CRS) {
        int  i_l = age_index_lower_CRS[i];
        int  i_u = age_index_upper_CRS[i];
        real t_l = age[i_l];
        real t_u = age[i_u];
        real patients = patient_years_CRS[i];
        expected_events_CRS[i] = patients * (H_EC_diagnosis[5,i_u] - H_EC_diagnosis[5,i_l]);
    }

    // Approximate cross-sectional distribution of AEH/EC
    // 1: Endometrial cancer (Stage I)
    // 2: Endometrial cancer (Stage II)
    // 3: Endometrial cancer (Stage III)
    // 4: Endometrial cancer (Stage IV)
    // 5: AEH
    vector[5] stage_dist_xs;
    {
        vector[5] E_t_if_visited;
        vector[5] p_visited;
        vector[5] E_t;

        E_t_if_visited[1] = 1 / (xi[2] + lambda[2]);
        E_t_if_visited[2] = 1 / (xi[3] + lambda[3]);
        E_t_if_visited[3] = 1 / (xi[4] + lambda[4]);
        E_t_if_visited[4] = 1 / xi[5];
        E_t_if_visited[5] = 1 / (rho + xi[1] + lambda[1]);

        p_visited[1] = lambda[1] / (rho + xi[1] + lambda[1]);
        p_visited[2] = p_visited[1] * lambda[2] / (xi[2] + lambda[2]);
        p_visited[3] = p_visited[2] * lambda[3] / (xi[3] + lambda[3]);
        p_visited[4] = p_visited[3] * lambda[4] / (xi[4] + lambda[4]);
        p_visited[5] = 1.0;

        E_t = E_t_if_visited .* p_visited;

        stage_dist_xs = E_t / sum(E_t);
    }

    // Endometrial cancer stage distribution
    vector[4] stage_dist_EC;
    {
        matrix[16,16] S = rep_matrix(0.0, 16, 16);
        
        S[1,2] = xi[1] / (rho + xi[1] + lambda[1]);
        S[1,4] = rho / (rho + xi[1] + lambda[1]);
        S[2,3] = delta[1] / (rho + delta[1] + lambda[1]);
        S[2,4] = rho / (rho + delta[1] + lambda[1]);
        S[3,3] = 1.0;
        S[4,4] = 1.0;

        S[1,5] = lambda[1] / (rho + xi[1] + lambda[1]);

        S[2,9] = lambda[1] / (rho + delta[1] + lambda[1]);

        S[5,6] = lambda[2] / (xi[2] + lambda[2]);
        S[6,7] = lambda[3] / (xi[3] + lambda[3]);
        S[7,8] = lambda[4] / (xi[4] + lambda[4]);

        S[5,9] = xi[2] / (xi[2] + lambda[2]);
        S[6,10] = xi[3] / (xi[3] + lambda[3]);
        S[7,11] = xi[4] / (xi[4] + lambda[4]);
        S[8,12] = 1.0;

        S[9,10] = lambda[2] / (delta[2] + lambda[2]);
        S[10,11] = lambda[3] / (delta[3] + lambda[3]);
        S[11,12] = lambda[4] / (delta[4] + lambda[4]);

        S[9,13] = delta[2] / (delta[2] + lambda[2]);
        S[10,14] = delta[3] / (delta[3] + lambda[3]);
        S[11,15] = delta[4] / (delta[4] + lambda[4]);
        S[12,16] = 1.0;

        S[13,13] = 1.0;
        S[14,14] = 1.0;
        S[15,15] = 1.0;
        S[16,16] = 1.0;

        row_vector[16] y0   = one_hot_row_vector(16, 1);
        matrix[16,16]  S6   = matrix_power(S, 6);

        row_vector[4] temp = (y0 * S6)[13:16];
        stage_dist_EC = temp' / sum(temp);
    }
}

model {

    // PRIORS
    for (g in 1:5) {
        for (i in 1:N_basis) {
            alpha[g,i] ~ normal(-10.0, 2.0);
        }
    }

    rho ~ lognormal(-2.0, 0.2);

    lambda[1] ~ lognormal(-2.6, 0.2);

    for (i in 2:4) {
        lambda[i] ~ lognormal(0.0, 1.0);
    }
    
    for (i in 1:5) {
        xi[i] ~ lognormal(0.0, 1.0);
        delta[i] ~ lognormal(0.0, 1.0);
    }
    

    // LIKELIHOOD
    for (g in 1:N_gt_PLSD)
        events_PLSD[g] ~ poisson(expected_events_PLSD[g]);

    events_CRS ~ poisson(expected_events_CRS);

    stage_dist_RenkonenSinisalo2007 ~ multinomial(stage_dist_xs);
    stage_dist_Eikenboom2021 ~ multinomial(stage_dist_xs);

    stage_dist_EC_England ~ multinomial(stage_dist_EC);

}

generated quantities {
    array[N_gt_PLSD,N_age_PLSD] int ppc_events_PLSD;
    array[N_CRS] int ppc_events_CRS;
    array[5] int ppc_stage_dist_RenkonenSinisalo2007;
    array[5] int ppc_stage_dist_Eikenboom2021;
    array[4] int ppc_stage_dist_EC_England;

    for (g in 1:N_gt_PLSD)
        ppc_events_PLSD[g] = poisson_rng(expected_events_PLSD[g]);

    ppc_events_CRS = poisson_rng(expected_events_CRS);
    ppc_stage_dist_RenkonenSinisalo2007 = multinomial_rng(stage_dist_xs, sum(stage_dist_RenkonenSinisalo2007));
    ppc_stage_dist_Eikenboom2021 = multinomial_rng(stage_dist_xs, sum(stage_dist_Eikenboom2021));
    ppc_stage_dist_EC_England = multinomial_rng(stage_dist_EC, sum(stage_dist_EC_England));
}