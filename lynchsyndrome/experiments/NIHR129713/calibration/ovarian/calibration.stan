functions {

    real bspline_basis(real x, row_vector extended_knots, int ind, int order);
    real bspline_basis(real x, row_vector extended_knots, int ind, int order) {
        real basis = 0.0;
        real w1 = 0.0;
        real w2 = 0.0;
        real b1 = 0.0;
        real b2 = 0.0;
        if ((order == 1) && (extended_knots[ind] <= x) && (x <= extended_knots[ind+1]))
            basis = 1.0;
        else {
            if (extended_knots[ind] != extended_knots[ind+order-1])
                w1 = (x - extended_knots[ind]) / (extended_knots[ind+order-1] - extended_knots[ind]);
            if (extended_knots[ind+1] != extended_knots[ind+order])
                w2 = 1.0 - (x - extended_knots[ind+1]) / (extended_knots[ind+order] - extended_knots[ind+1]);
            if (w1 > 0.0)
                b1 = bspline_basis(x, extended_knots, ind, order-1);
            if (w2 > 0.0)
                b2 = bspline_basis(x, extended_knots, ind+1, order-1);
            basis = w1 * b1 + w2 * b2;
        }
        return basis;
    }

    /**
     * Calculate a B-spline basis at a given point, including the intercept
     *
     * @param  x               the value at which to evaluate the B-Spline
     * @param  extended_knots  the knots, plus the boundary knots, plus
     *                           the boundary knots repeated at each end p
     *                           times, where p+1 is the order of the B-spline
     * @param  order           the order of the B-spline, i.e., the polynomial
     *                           degree plus one
     */
    row_vector bspline(real x, row_vector extended_knots, int order) {
        int p = order - 1;
        int q = num_elements(extended_knots) - 2 * p;
        int n = p + (q - 1);
        row_vector[n] res;
        for (i in 1:n) {
            res[i] = bspline_basis(x, extended_knots, i, order);
        }
        return res;
    }

    vector dydt(real       t,
                vector     state,
                row_vector extended_knots,
                int        order,
                vector     alpha,
                vector     lambda,
                vector     xi) {
        matrix[9,9] A = rep_matrix(0.0, 9, 9);

        A[2,1] = exp(bspline(t, extended_knots, order) * alpha);
        A[1,1] = -A[2,1];

        A[2,2] = -(lambda[1] + xi[1]);
        A[3,2] = lambda[1];
        A[3,3] = -(lambda[2] + xi[2]);
        A[4,3] = lambda[2];
        A[4,4] = -(lambda[3] + xi[3]);
        A[5,4] = lambda[3];
        A[5,5] = -xi[4];
        A[6,2] = xi[1];
        A[7,3] = xi[2];
        A[8,4] = xi[3];
        A[9,5] = xi[4];

        return A * state;
    }

    vector forecast(
        vector phi,
        vector theta,
        data array[] real x_r,
        data array[] int x_i
    ) {
        // Extract parameters

        // x_i tells us some key dimensions
        int N_age = x_i[1];
        int bs_degree = x_i[2];
        int n_basis = x_i[3];
        int bs_num_extended_knots = x_i[4];
        
        // x_r gives us the spline knot positions and the age array
        row_vector[bs_num_extended_knots] bs_extended_knots = to_row_vector(x_r[1:bs_num_extended_knots]);
        array[N_age] real age = x_r[(bs_num_extended_knots+1):(bs_num_extended_knots+N_age)];

        // phi are shared parameters which apply across all elements
        vector[4] xi = phi[1:4];

        // theta are specific parameters
        vector[n_basis] alpha = theta[1:n_basis];
        vector[3] lambda = theta[(n_basis+1):(n_basis+3)];

        // Calculate
        vector[9] y0 = one_hot_vector(9, 1);
        array[N_age] vector[9] ys;
        ys = ode_bdf(dydt, y0, 0.0, age, bs_extended_knots, bs_degree+1, alpha, lambda, xi);

        vector[N_age*5] res;
        for (i in 1:N_age) {
            int s = (i-1) * 5;
            res[s+1] = ys[i,6];
            res[s+2] = ys[i,7];
            res[s+3] = ys[i,8];
            res[s+4] = ys[i,9];
            res[s+5] = sum(ys[i,1:5]);
        }

        return res;
    }

}

data {
    int<lower=0> bs_degree;
    int<lower=2> bs_num_knots;
    row_vector[bs_num_knots] bs_knots;

    int<lower=1> N_age;
    array[N_age] real age;

    // Prospective Lynch Syndrome Database
    int<lower=1> N_PLSD;
    int<lower=1> N_gt_PLSD;
    array[N_PLSD]           int  age_index_lower_PLSD;
    array[N_PLSD]           int  age_index_upper_PLSD;
    array[N_gt_PLSD,N_PLSD] real patient_years_PLSD;
    array[N_gt_PLSD,N_PLSD] int  events_PLSD;

    // Cancer Registration Statistics, England (2017)
    int<lower=1> N_CRS;
    array[N_CRS] int  age_index_lower_CRS;
    array[N_CRS] int  age_index_upper_CRS;
    array[N_CRS] real patient_years_CRS;
    array[N_CRS] int  events_CRS;

    // Stage distribution
    array[4] int stage_dist_OC_England;
    array[4] int stage_dist_Woolderink2018;
    array[4] int stage_dist_Duenas2020;
}

transformed data {
    int n_basis = bs_num_knots + bs_degree - 1;
    int bs_num_extended_knots = bs_num_knots + 2 * bs_degree;
    row_vector[bs_num_extended_knots] bs_extended_knots;
    bs_extended_knots[(bs_degree+1):(bs_degree+bs_num_knots)] = bs_knots;
    if (bs_degree > 1) {
        bs_extended_knots[1:bs_degree] = rep_row_vector(bs_knots[1], bs_degree);
        bs_extended_knots[(bs_num_knots+bs_degree+1):bs_num_extended_knots] = rep_row_vector(bs_knots[bs_num_knots], bs_degree);
    }

    array[5,(bs_num_extended_knots+N_age)] real x_r;
    array[5,4] int  x_i;

    {
        array[bs_num_extended_knots+N_age] real x_rg;
        for (j in 1:bs_num_extended_knots)
            x_rg[j] = bs_extended_knots[j];
        for (j in 1:N_age)
            x_rg[bs_num_extended_knots+j] = age[j];
        x_r = { x_rg, x_rg, x_rg, x_rg, x_rg };
    }

    for (g in 1:5) {
        x_i[g] = {N_age, bs_degree, n_basis, bs_num_extended_knots};
    }
}

parameters {
    array[5] vector[n_basis] alpha;
    vector<lower=0>[3] lambda_LS;
    vector<lower=0>[3] lambda_sporadic;
    positive_ordered[4] xi;
}

transformed parameters {
    array[5,N_age]          vector[4] F_OC_stage;
    array[5,N_age]          real      S_OC;
    array[N_gt_PLSD,N_PLSD] real      expected_events_PLSD;
    array[N_CRS]            real      expected_events_CRS;
    
    vector[4] stage_dist_LS;
    vector[4] stage_dist_sporadic;

    {
        vector[N_age * 5 * 5] map_rect_res;
        vector[4] phi = xi;
        array[5] vector[n_basis+3] theta;
        for (g in 1:4) {
            theta[g] = append_row(alpha[g], lambda_LS);
        }
        theta[5] = append_row(alpha[5], lambda_sporadic);

        map_rect_res = map_rect(forecast, phi, theta, x_r, x_i);

        for (g in 1:5) {
            for (i in 1:N_age) {
                int s = (g-1) * (N_age*5) + (i-1)*5;
                F_OC_stage[g,i] = map_rect_res[(s+1):(s+4)];
                S_OC[g,i] = map_rect_res[s+5];
            }
        }

        for (g in 1:N_gt_PLSD) {
            for (i in 1:N_PLSD) {
                real t_l = age[age_index_lower_PLSD[i]];
                real t_r = age[age_index_upper_PLSD[i]];
                real S_l = S_OC[g,age_index_lower_PLSD[i]];
                real S_r = S_OC[g,age_index_upper_PLSD[i]];
                real rate = 1e-10;
                if (S_l > S_r && S_r > 0.0)
                    rate = (log(S_l) - log(S_r)) / (t_r - t_l);
                expected_events_PLSD[g,i] = rate * patient_years_PLSD[g,i];
            }
        }

        for (i in 1:N_CRS) {
            real t_l = age[age_index_lower_CRS[i]];
            real t_r = age[age_index_upper_CRS[i]];
            real S_l = S_OC[5,age_index_lower_CRS[i]];
            real S_r = S_OC[5,age_index_upper_CRS[i]];
            real rate = 1e-16;
            if (S_l > S_r && S_r > 0.0)
                rate = (log(S_l) - log(S_r)) / (t_r - t_l);
            expected_events_CRS[i] = rate * patient_years_CRS[i];
        }

        vector[4] rho_LS;
        rho_LS[1] = xi[1] / (xi[1] + lambda_LS[1]);
        rho_LS[2] = lambda_LS[1] / (xi[1] + lambda_LS[1]) * xi[2] / (xi[2] + lambda_LS[2]);
        rho_LS[3] = lambda_LS[1] / (xi[1] + lambda_LS[1]) * lambda_LS[2] / (xi[2] + lambda_LS[2]) * xi[3] / (xi[3] + lambda_LS[3]);
        rho_LS[4] = lambda_LS[1] / (xi[1] + lambda_LS[1]) * lambda_LS[2] / (xi[2] + lambda_LS[2]) * lambda_LS[3] / (xi[3] + lambda_LS[3]);
        stage_dist_LS = rho_LS / sum(rho_LS);

        vector[4] rho_sporadic;
        rho_sporadic[1] = xi[1] / (xi[1] + lambda_sporadic[1]);
        rho_sporadic[2] = lambda_sporadic[1] / (xi[1] + lambda_sporadic[1]) * xi[2] / (xi[2] + lambda_sporadic[2]);
        rho_sporadic[3] = lambda_sporadic[1] / (xi[1] + lambda_sporadic[1]) * lambda_sporadic[2] / (xi[2] + lambda_sporadic[2]) * xi[3] / (xi[3] + lambda_sporadic[3]);
        rho_sporadic[4] = lambda_sporadic[1] / (xi[1] + lambda_sporadic[1]) * lambda_sporadic[2] / (xi[2] + lambda_sporadic[2]) * lambda_sporadic[3] / (xi[3] + lambda_sporadic[3]);
        stage_dist_sporadic = rho_sporadic / sum(rho_sporadic);
    }
}

model {
    // PRIORS
    for (g in 1:5)
        alpha[g] ~ normal(-20.0, 2.0);
    lambda_LS ~ lognormal(0.0, 1.0);
    lambda_sporadic ~ lognormal(0.0, 1.0);
    xi ~ lognormal(0.0, 1.0);

    // LIKELIHOOD
    for (g in 1:N_gt_PLSD)
        events_PLSD[g] ~ poisson(expected_events_PLSD[g]);

    events_CRS ~ poisson(expected_events_CRS);

    stage_dist_OC_England ~ multinomial(stage_dist_sporadic);

    stage_dist_Woolderink2018 ~ multinomial(stage_dist_LS);
    stage_dist_Duenas2020 ~ multinomial(stage_dist_LS);
}

generated quantities {
    // vector[101] age = linspaced_vector(101, 0.0, 100.0);
    // matrix[101,bs_num_knots+bs_degree-1] age_basis;
    // for (i in 1:101) {
    //     age_basis[i] = bspline(age[i], bs_extended_knots, bs_degree+1);
    // }
}