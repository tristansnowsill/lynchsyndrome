version 17.0

capture log close

log using "\\wsl$\Ubuntu\home\tms211\Projects\lynchsyndrome\lynchsyndrome\experiments\NIHR129713\calibration\colorectal\survival.smcl", replace

use "\\wsl$\Ubuntu\home\tms211\Projects\lynchsyndrome\lynchsyndrome\experiments\NIHR129713\calibration\colorectal\ons2018-crc.dta", clear

capture program drop crcsurv
program crcsurv
	// Suggested to call as follows
	//
	// . ml model lf crcsurv (t s s_se = [lambda_depvars]) ([theta_depvars])
	//
	// Where t, s and s_se are the time point, survival estimate, and standard
	// error of the survival estimate, lambda_depvars are the variables
	// expected to have a linear effect on the log hazard rate, and
	// theta_depvars are the variables expected to have a linear effect on the
	// log frailty variance.
	//
	// Most likely the call will be
	//
	// . ml model lf crcsurv (t s s_se = i.stage i.sex i.agegrp) ()
	//
	version 17.0
	args lnf ln_lambda ln_theta
	tempvar lambda theta s_pred
	quietly generate double `lambda' = exp(`ln_lambda')
	quietly generate double `theta' = exp(`ln_theta')
	quietly generate double `s_pred' = (1 + `lambda' * `theta' * $ML_y1) ^ (-1/`theta')
	quietly replace `lnf' = lnnormalden(`s_pred', $ML_y2, $ML_y3)
end

ml model lf crcsurv (years surv surv_se = i.stage i.sex ib4.agegrp) ()
ml check
ml search
ml maximize

estat vce


log close