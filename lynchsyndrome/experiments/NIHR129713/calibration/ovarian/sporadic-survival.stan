data {
    vector[4] surv_1y;
    vector[4] surv_1y_lcl;
    vector[4] surv_1y_ucl;

    vector[4] surv_5y;
    vector[4] surv_5y_lcl;
    vector[4] surv_5y_ucl;
}

transformed data {
    vector[4] se_surv_1y;
    vector[4] se_surv_5y;

    for (i in 1:4) {
        se_surv_1y[i] = (surv_1y_ucl[i] - surv_1y_lcl[i]) / 3.92;
        se_surv_5y[i] = (surv_5y_ucl[i] - surv_5y_lcl[i]) / 3.92;
    }
}

parameters {
    vector<lower=0>[4] lambda;
    real<lower=0>      delta;
    real<lower=0>      theta;
}

transformed parameters {
    vector[4] model_surv_1y;
    vector[4] model_surv_5y;

    {
        for (i in 1:4) {
            real nonfrailty_1y;
            real nonfrailty_5y;
            if (lambda[i] == delta) {
                nonfrailty_1y = exp(-lambda[i]) * (1 + lambda[i]);
                nonfrailty_5y = exp(-lambda[i] * 5.0) * (1 + lambda[i] * 5.0);
            } else {
                nonfrailty_1y = (delta*exp(-lambda[i]) - lambda[i]*exp(-delta)) / (delta - lambda[i]);
                nonfrailty_5y = (delta*exp(-lambda[i]*5) - lambda[i]*exp(-delta*5)) / (delta - lambda[i]);
            }
            model_surv_1y[i] = (1 - theta * log(nonfrailty_1y)) ^ (-1/theta);
            model_surv_5y[i] = (1 - theta * log(nonfrailty_5y)) ^ (-1/theta);
        }

    }
}

model {
    // PRIORS
    lambda ~ lognormal(0, 1);
    delta  ~ lognormal(0, 1);
    theta  ~ lognormal(0, 1);

    // LIKELIHOOD
    surv_1y ~ normal(model_surv_1y, se_surv_1y);
    surv_5y ~ normal(model_surv_5y, se_surv_5y);
}