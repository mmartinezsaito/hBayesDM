#include /pre/license.stan

// based on choiceRT_ddm.stan in the hBayesDM R package 
data {
  int<lower=1> N;      // Number of subjects
  int<lower=0> Nu_max; // Max (across subjects) number of upper boundary responses
  int<lower=0> Nl_max; // Max (across subjects) number of lower boundary responses
  int<lower=0> Nu[N];  // Number of upper boundary responses for each subj
  int<lower=0> Nl[N];  // Number of lower boundary responses for each subj
  real RTu[N, Nu_max];  // upper boundary response times
  real RTl[N, Nl_max];  // lower boundary response times
  real minRT[N];       // minimum RT for each subject of the observed data
  real RTbound;        // lower bound or RT across all subjects (e.g., 0.1 second)

  int<lower=0> NC_max; // Max (across subjects and confidence judgments) number of responses
  int<lower=0> Nu1[N];  // Number of upper confidence level 1 responses
  int<lower=0> Nu2[N];  // Number of upper confidence level 2 responses
  int<lower=0> Nu3[N];  // Number of upper confidence level 3 responses
  int<lower=0> Nu4[N];  // Number of upper confidence level 4 responses
  int<lower=0> Nl1[N];  // Number of lower confidence level 1 responses
  int<lower=0> Nl2[N];  // Number of lower confidence level 2 responses
  int<lower=0> Nl3[N];  // Number of lower confidence level 3 responses
  int<lower=0> Nl4[N];  // Number of lower confidence level 4 responses
  real RTu1[N, NC_max];  // upper boundary confidence judgment level 1 times
  real RTu2[N, NC_max];  // upper boundary confidence judgment level 2 times
  real RTu3[N, NC_max];  // upper boundary confidence judgment level 3 times
  real RTu4[N, NC_max];  // upper boundary confidence judgment level 4 times
  real RTl1[N, NC_max];  // lower boundary confidence judgment level 1 times
  real RTl2[N, NC_max];  // lower boundary confidence judgment level 2 times
  real RTl3[N, NC_max];  // lower boundary confidence judgment level 3 times
  real RTl4[N, NC_max];  // lower boundary confidence judgment level 4 times
  real minCT[N];       // minimum confidence RT for each subject of the observed data
}

parameters {
  // Parameters of the DDM (parameter names in Ratcliffs DDM), from https://github.com/gbiele/stan_wiener_test/blob/master/stan_wiener_test.R
  //   alpha (a): Boundary separation or Speed-accuracy trade-off (high alpha means high accuracy). alpha > 0
  //   beta (b): Initial bias Bias for either response (beta > 0.5 means bias towards "upper" response 'A'). 0 < beta < 1
  //   delta (v): Drift rate Quality of the stimulus (delta close to 0 means ambiguous stimulus or weak ability). 0 < delta
  //   tau (ter): Nondecision time + Motor response time + encoding time (high means slow encoding, execution). 0 < ter (in seconds)
  //   /* IMPORTANT: upper boundary of tau must be smaller than minimum RT to avoid zero likelihood for fast responses.
  //                 tau cannot for physiological reasons be faster than 0.1s */
  // Parameters of the 2DSD (Pleskac & Busemeyer, 2011) not in Ratcliff's DDM
  //   alpha1, alpha2, alpha3, alpha4: boundaris of the 4 (retrospective) confidence rating levels
  //   jump: perceptual decision-induced leap in the evidence accumulation process (respect to the boundary alpha) 
  //   delta2: drift rate after the 1st order judgment (choice) but before the 2nd order judgment
  //   tau2: Nondecision time + Motor response time + encoding time (high means slow encoding, execution). 0 < tau2 (in seconds)

  // Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[9] mu_pr;
  vector<lower=0>[9] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] alpha_pr;
  vector[N] beta_pr;
  vector[N] delta_pr;
  vector[N] tau_pr;
  vector[N] alpha1_pr;
  vector[N] cr_unit_pr;
  vector[N] jump_pr;
  vector[N] delta2_pr;
  vector[N] tau2_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0>[N]                          alpha; // boundary separation
  vector<lower=0, upper=1>[N]                 beta;  // initial bias
  vector<lower=0>[N]                          delta; // drift rate
  vector<lower=RTbound, upper=max(minRT)>[N]  tau; // nondecision time
  vector<lower=0>[N]                          alpha1;
  vector<lower=0>[N]                          cr_unit;
  vector<lower=0>[N]                          alpha4; 
  vector<lower=0, upper=1>[N]                 jump;
  vector<lower=0>[N]                          delta2;
  vector<lower=RTbound, upper=max(minCT)>[N]  tau2; // nondecision time  

  for (i in 1:N) {
    beta[i] = Phi_approx(mu_pr[2] + sigma[2] * beta_pr[i]);
    tau[i]  = Phi_approx(mu_pr[4] + sigma[4] * tau_pr[i]) * (minRT[i] -0.001 - RTbound) + RTbound;
    jump[i]  = Phi_approx(mu_pr[7] + sigma[7] * jump_pr[i]);
    tau2[i]  = Phi_approx(mu_pr[9] + sigma[9] * tau2_pr[i]) * (minCT[i] -0.001 - RTbound) + RTbound; // to avoid wiener complaints
  }
  alpha = exp(mu_pr[1] + sigma[1] * alpha_pr);
  delta = exp(mu_pr[3] + sigma[3] * delta_pr);
  alpha1 = exp(mu_pr[5] + sigma[5] * alpha1_pr);
  cr_unit = exp(mu_pr[6] + sigma[6] * cr_unit_pr);
  alpha4 = 3 * cr_unit + alpha1;
  delta2 = exp(mu_pr[8] + sigma[8] * delta2_pr);
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  // Individual parameters for non-centered parameterization
  alpha_pr ~ normal(0, 1);
  beta_pr  ~ normal(0, 1);
  delta_pr ~ normal(0, 1);
  tau_pr   ~ normal(0, 1);
  alpha1_pr ~ normal(0, 1);
  cr_unit_pr ~ normal(0, 1);
  jump_pr ~ normal(0, 1);
  delta2_pr ~ normal(0, 1);
  tau2_pr   ~ normal(0, 1);
  
  // maybe the first-hitting time model application to 4 CR levels can be explained with the jump-tau2 combination
  // Begin subject loop
  for (i in 1:N) {
    // Response time distributed along wiener first passage time distribution
    RTu[i, :Nu[i]] ~ wiener(alpha[i], tau[i], beta[i], delta[i]);
    RTl[i, :Nl[i]] ~ wiener(alpha[i], tau[i], 1-beta[i], -delta[i]);

    RTu1[i, :Nu1[i]] ~ wiener(alpha1[i],              tau2[i], jump[i],   delta2[i]);
    RTu2[i, :Nu2[i]] ~ wiener(alpha1[i] + cr_unit[i], tau2[i], jump[i],   delta2[i]);
    RTu3[i, :Nu3[i]] ~ wiener(alpha4[i] - cr_unit[i], tau2[i], jump[i],   delta2[i]);
    RTu4[i, :Nu4[i]] ~ wiener(alpha4[i],              tau2[i], jump[i],   delta2[i]);
    RTl1[i, :Nl1[i]] ~ wiener(alpha1[i],              tau2[i], 1-jump[i], -delta2[i]);
    RTl2[i, :Nl2[i]] ~ wiener(alpha1[i] + cr_unit[i], tau2[i], 1-jump[i], -delta2[i]);
    RTl3[i, :Nl3[i]] ~ wiener(alpha4[i] - cr_unit[i], tau2[i], 1-jump[i], -delta2[i]);
    RTl4[i, :Nl4[i]] ~ wiener(alpha4[i],              tau2[i], 1-jump[i], -delta2[i]);
  } // end of subject loop
}

generated quantities {
  // For group level parameters
  real<lower=0>           mu_alpha; // boundary separation
  real<lower=0, upper=1>  mu_beta;  // initial bias
  real<lower=0>           mu_delta; // drift rate
  real<lower=RTbound, upper=max(minRT)> mu_tau; // nondecision time
  real<lower=0>           mu_alpha1; 
  real<lower=0>           mu_alpha4;
  real<lower=0, upper=1> mu_jump; 
  real<lower=0>           mu_delta2;
  real<lower=RTbound, upper=max(minCT)> mu_tau2; 
  real<lower=0>           mu_cr_unit;
  real<lower=0>           mu_cr1;
  real<lower=0>           mu_cr4;
  
  // For log likelihood calculation
  real log_lik[N];

  // Assign group level parameter values
  mu_alpha  = exp(mu_pr[1]);
  mu_beta   = Phi_approx(mu_pr[2]);
  mu_delta  = exp(mu_pr[3]);
  mu_tau    = Phi_approx(mu_pr[4]) * (mean(minRT)-RTbound) + RTbound;
  mu_alpha1 = exp(mu_pr[5]);
  mu_cr_unit= exp(mu_pr[6]);
  mu_jump   = Phi_approx(mu_pr[7]);
  mu_delta2 = exp(mu_pr[8]);
  mu_tau2   = Phi_approx(mu_pr[9]) * (mean(minCT)-RTbound) + RTbound;
  
  mu_alpha4 = 3 * mu_cr_unit + mu_alpha1;
  mu_cr1 = mu_alpha + mu_alpha1;
  mu_cr4 = mu_alpha + mu_alpha4;  

  { // local section, this saves time and space
    // Begin subject loop
    for (i in 1:N) {
      log_lik[i]  = wiener_lpdf(RTu[i, :Nu[i]] | alpha[i], tau[i], beta[i], delta[i]);
      log_lik[i] += wiener_lpdf(RTl[i, :Nl[i]] | alpha[i], tau[i], 1-beta[i], -delta[i]);
      
      log_lik[i] += wiener_lpdf(RTu1[i, :Nu1[i]] | alpha1[i], tau2[i], jump[i], delta2[i]);
      log_lik[i] += wiener_lpdf(RTu2[i, :Nu2[i]] | alpha1[i] + cr_unit[i], tau2[i], jump[i], delta2[i]);
      log_lik[i] += wiener_lpdf(RTu3[i, :Nu3[i]] | alpha4[i] - cr_unit[i], tau2[i], jump[i], delta2[i]);
      log_lik[i] += wiener_lpdf(RTu4[i, :Nu4[i]] | alpha4[i], tau2[i], jump[i], delta2[i]);
      log_lik[i] += wiener_lpdf(RTl1[i, :Nl1[i]] | alpha1[i], tau2[i], 1-jump[i], -delta2[i]);
      log_lik[i] += wiener_lpdf(RTl2[i, :Nl2[i]] | alpha1[i] + cr_unit[i], tau2[i], 1-jump[i], -delta2[i]);
      log_lik[i] += wiener_lpdf(RTl3[i, :Nl3[i]] | alpha4[i] - cr_unit[i], tau2[i], 1-jump[i], -delta2[i]);
      log_lik[i] += wiener_lpdf(RTl4[i, :Nl4[i]] | alpha4[i], tau2[i], 1-jump[i], -delta2[i]);
    }
  }
}


