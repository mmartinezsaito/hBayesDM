#include /pre/license.stan

data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int<lower=0,upper=1> choice[N,T];
  int<lower=0,upper=1> outcome[N,T];
  int<lower=0,upper=9> cue[N,T];
}

transformed data {
  real sigmaO; 
  vector<lower=0, upper=1>[9] Pwc0;  // probability of winning given the cue according to the briefing
  vector<lower=0, upper=1>[9] Pwca;  // probability of winning given the actual winning densities

  sigmaO = .04;  // observation noise

  Pwc0 = [0, .125, .25, .375, .5, .625, .75, .875, 1]';     
  Pwca = [0, .6, .6, .6, .6, .6, .6, .6, 1]';

   //Pwc1 = [0, p1, p1, p1, p1, p1, p1, p1, 1]';  // probabilities of winning given the cue if a fixed value is assumed 
   //Pwc = (1 - sw) * Pwc0 + sw * Pwc1; 
   //Pwc = (1 - sw) * Pwc0 + sw * Pwca; 

}

parameters {
  // group-level parameters
  vector[6]          mu_gp;
  vector<lower=0>[6] sigma_gp;

  // subject-level raw parameters, follows norm(0,1), for later Matt Trick
  vector[N] beta_raw;      // inverse softmax temperature
  vector[N] sigma0_raw;    // GAME PRIOR: anticipated initial sd of the 9 cards
  vector[N] sigmaD_raw;    // sd of diffusion noise

  // decay factor: it weights the contribution of the alternative (actual) model against the default prior modeli by a decay to the default model
  vector[N] lambdaU_raw;
  vector[N] lambdaK_raw;
  vector[N] lambda0_raw; 

   //real p1;          // probability of winning given the cue if a fixed value, not necessarily 0.6, is assumed  <-------- COMMENT IN/OUT 

}

transformed parameters {
  // subject-level parameters
  vector<lower=0,upper=1>[N]   beta;
  vector<lower=0,upper=.15>[N] sigma0;
  vector<lower=0,upper=.15>[N] sigmaD;
  vector<lower=0,upper=1>[N]   lambdaU;
  vector<lower=0>[N]           lambdaK;
  vector<lower=0,upper=T>[N]   lambda0;

  // Matt Trick
  // the (group) hyperparameters Gaussian location and scale constrain the individual parameters
  for (i in 1:N) {
    beta[i]    = Phi_approx( mu_gp[1] + sigma_gp[1] * beta_raw[i] );
    sigma0[i]  = Phi_approx( mu_gp[2] + sigma_gp[2] * sigma0_raw[i] ) * .15;
    sigmaD[i]  = Phi_approx( mu_gp[3] + sigma_gp[3] * sigmaD_raw[i] ) * .15;
    lambdaU[i] = Phi_approx( mu_gp[4] + sigma_gp[4] * lambdaU_raw[i] );
    lambdaK[i] = exp(        mu_gp[5] + sigma_gp[5] * lambdaK_raw[i] );
    lambda0[i] = Phi_approx( mu_gp[6] + sigma_gp[6] * lambdaK_raw[i] ) * T;
  }
}

model {

  // ======= BAYESIAN PRIORS ======= //
  // group level hyperparameters
  mu_gp    ~ normal(0,1);
  sigma_gp ~ cauchy(0,5); 

  // individual parameters: non-centered parameterization
  beta_raw    ~ normal(0,1);  
  sigma0_raw  ~ normal(0,1);  // game prior: cognitive flexibility for alternative model search
  sigmaD_raw  ~ normal(0,1);
  lambdaU_raw ~ std_normal();
  lambdaK_raw ~ std_normal();
  lambda0_raw ~ std_normal();
   //p1       ~ std_normal();


  // ======= LIKELIHOOD FUNCTION ======= //
  // subject loop and trial loop
  for (i in 1:N) {
    vector[9] mu_est;    // estimated mean for each option 
    vector[9] var_est;   // estimated sd^2 for each option
    real pe;             // prediction error
    real k;              // learning rate

    mu_est  = Pwc0;
    var_est = rep_vector(sigma0[i]^2, 9);

    for (t in 1:(Tsubj[i])) {
      int q;
      q = cue[i, t];

      if (q != 1 && q != 9) { 
        real lambda;

        // compute action probabilities
         // choice[i,t] ~ categorical_logit(1, beta[i] * mu_est );
        choice[i, t] ~ bernoulli_logit(beta[i] * mu_est[q]);     // for beta model (beta[i] * Pwc[cue[i, t]);  

        // --- Update --- //
        // prediction error: innovation (pre-fit residual) measurement
        pe = outcome[i,t] - mu_est[q];
        // innovation (pre-fit residual) covariance is just var_est[] + sigma0^2

        // learning rate: optimal Kalman gain
        k = var_est[q] / ( var_est[q] + sigmaO^2 );

        // value updating (learning)
        mu_est[q] += k * pe;   // updated state estimate
        var_est[q] *= (1 - k); // updated covariance estimate

        // --- Predict --- // 
        // lambda weighs the two models by decay, and it's a sigmoid of time 
        lambda = lambdaU[i] * logistic_cdf(t , lambda0[i], 1 / lambdaK[i]);
        // diffusion process 
        mu_est  = lambda * mu_est + (1 - lambda) * Pwc0[q]; 
        var_est = lambda^2 * var_est + sigmaD[i]^2;
      }
    }
  }

}

generated quantities {
  real<lower=0,upper=1>   mu_beta;
  real<lower=0,upper=.15> mu_sigma0;
  real<lower=0,upper=.15> mu_sigmaD;
  real<lower=0,upper=1>   mu_lambdaU;
  real<lower=0>           mu_lambdaK;
  real<lower=0,upper=T>   mu_lambda0;

  real log_lik[N];

  real winp_mean[N, T];
  real winp_var[N, T];
  real decay[N, T];
  real rpe[N, T];

  real y_pred[N,T];
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_beta    = Phi_approx(mu_gp[1]);
  mu_sigma0  = Phi_approx(mu_gp[2]) * .15;
  mu_sigmaD  = Phi_approx(mu_gp[3]) * .15;
  mu_lambdaU = Phi_approx(mu_gp[4]);
  mu_lambdaK = exp(mu_gp[5]);
  mu_lambda0 = Phi_approx(mu_gp[6]) * T;

  // subject and trial loop
  for (i in 1:N) {
    vector[9] mu_est;    // estimated mean for each option 
    vector[9] var_est;   // estimated sd^2 for each option
    real pe;             // prediction error
    real k;              // learning rate

    mu_est  = Pwc0;
    var_est = rep_vector(sigma0[i]^2, 9);

    log_lik[i] = 0;  // ------- GQ ------

    for (t in 1:(Tsubj[i])) {
      int q;
      real lambda;

      q = cue[i, t];

      // lambda weighs the two models by decay, and it's a sigmoid of time 
      lambda = lambdaU[i] * logistic_cdf(t , lambda0[i], 1 / lambdaK[i]);

      if (q != 1 && q != 9) { 

        // --- Update --- //
        // prediction error: innovation (pre-fit residual) measurement
        pe = outcome[i,t] - mu_est[q];
        // innovation (pre-fit residual) covariance is just var_est[] + sigma0^2

        // learning rate: optimal Kalman gain
        k = var_est[q] / ( var_est[q] + sigmaO^2 );

        // ------- GQ ------- //
        // compute action probabilities
        log_lik[i] += bernoulli_logit_lpmf(choice[i,t] | beta[i] * mu_est[q]);

        // generate posterior prediction for current trial
        y_pred[i, t] = bernoulli_logit_rng(beta[i] * mu_est[q]);

        // Model regressors: stored values before being updated
        winp_mean[i, t] = mu_est[q];
        winp_var[i, t] = var_est[q];
        decay[i, t] = lambda;
        rpe[i, t] = pe;
        // ------ GQ END ------ //

        // value updating (learning)
        mu_est[q] += k * pe;   // updated state estimate
        var_est[q] *= (1 - k); // updated covariance estimate

        // --- Predict --- // 
        // diffusion process 
        mu_est  = lambda * mu_est + (1 - lambda) * Pwc0[q]; 
        var_est = lambda^2 * var_est + sigmaD[i]^2;

      } else {

        winp_mean[i, t] = Pwc0[q];
        winp_var[i, t] = sigma0[i]^2;
        decay[i, t] = lambda;
        rpe[i, t] = outcome[i,t] - Pwc0[q];
      }
    }
  }
    
}
