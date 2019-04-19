#include /pre/license.stan

data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int<lower=0,upper=1> choice[N,T];
  int<lower=0,upper=1> outcome[N,T];
  int<lower=1,upper=9> cue[N,T];
}

transformed data {
  vector<lower=0, upper=1>[9] Pwc0;  // probability of winning given the cue according to the briefing
  vector<lower=0, upper=1>[9] Pwca;  // probability of winning given the actual winning densities

  Pwc0 = [0, .125, .25, .375, .5, .625, .75, .875, 1]';     
  Pwca = [0, .6, .6, .6, .6, .6, .6, .6, 1]';

}

parameters {
  // group-level parameters
  vector[2]          mu_nc;
  vector<lower=0>[2] sd_nc;

  // Fixed effect (yoked) parameters
  real<lower=0> sigmaO;       // observation noise 
  real<lower=0> sigmaE;      // evolution noise

  // subject-level raw parameters, follows norm(0,1), for later Matt Trick
  vector[N] sigma0_nc;    // GAME PRIOR: anticipated initial sd of the 9 cards

  // this weights the contribution of the alternative (actual) model against the default prior model
  vector[N] w_nc;

}

transformed parameters {
  // subject-level parameters
  vector<lower=0>[N]         sigma0;
  vector<lower=0,upper=1>[N] w;

  // Matt Trick
  // the (group) hyperparameters Gaussian location and scale constrain the individual parameters
  for (i in 1:N) {
    sigma0[i] = exp(        mu_nc[1] + sd_nc[1] * sigma0_nc[i] );
    w[i]      = Phi_approx( mu_nc[2] + sd_nc[2] * w_nc[i] );
  }
}

model {

  // ======= BAYESIAN PRIORS ======= //
  // group level hyperparameters
  mu_nc ~ normal(0,1);
  sd_nc ~ normal(0,1); 

  // individual fixed effects
  sigmaO ~ normal(0, 1);
  sigmaE ~ normal(0, 1);  

  // individual parameters: non-centered parameterization
  sigma0_nc ~ normal(0,1);  // game prior: cognitive flexibility for alternative model search
  w_nc      ~ std_normal();


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

        // compute action probabilities
         // choice[i,t] ~ categorical_logit( beta[i] * mu_est );
        choice[i, t] ~ bernoulli( mu_est[q] * w[i] + Pwc0[q] * (1-w[i]) );  

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
        // driftless diffusion 
        //mu_est  = mu_est;
        var_est += sigmaE^2;
      }
    }
  }

}

generated quantities {
  real<lower=0>         mu_sigma0;
  real<lower=0,upper=1> mu_w;
  real<lower=0>         sd_sigma0;
  real<lower=0>         sd_w;

  real log_lik[N];

  real winp_mean[N, T];
  real winp_var[N, T];
  real gain[N, T];
  real rpe[N, T];

  real y_pred[N,T];
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_sigma0  = exp(        mu_nc[1] );
  mu_w       = Phi_approx( mu_nc[2] );
  sd_sigma0  = exp(        sd_nc[1] );
  sd_w       = Phi_approx( sd_nc[2] );

  // subject and trial loop
  for (i in 1:N) {
    vector[9] mu_est;    // estimated mean for each option 
    vector[9] var_est;   // estimated sd^2 for each option
    real pe;             // prediction error
    real k;              // learning rate

    mu_est  = Pwc0;
    var_est = rep_vector(sigma0[i]^2, 9);

    // ------- GQ ------- //
    log_lik[i] = 0;
    // ------- END GQ ------- //

    for (t in 1:(Tsubj[i])) {
      int q;
      q = cue[i, t];

      // ------- GQ ------- //
      y_pred[i, t] = bernoulli_rng(mu_est[q] * w[i] + Pwc0[q] * (1-w[i])); // generate posterior prediction for current trial
      // ------- END GQ ------- //

      // --- Update --- //
      // prediction error: innovation (pre-fit residual) measurement
      pe = outcome[i,t] - mu_est[q];
      // innovation (pre-fit residual) covariance is just var_est[] + sigma0^2

      // learning rate: optimal Kalman gain
      k = var_est[q] / ( var_est[q] + sigmaO^2 );

      if (q != 1 && q != 9) { 

        // ------- GQ ------- //
        // compute action probabilities
        log_lik[i] += bernoulli_lpmf(choice[i,t] | mu_est[q] * w[i] + Pwc0[q] * (1-w[i]) );

        // Model regressors: stored values before being updated
        winp_mean[i, t] = mu_est[q];
        winp_var[i, t]  = var_est[q];
        gain[i, t]      = k;
        rpe[i, t]       = pe;
        // ------ GQ END ------ //

        // value updating (learning)
        mu_est[q] += k * pe;   // updated state estimate
        var_est[q] *= (1 - k); // updated covariance estimate

        // --- Predict --- // 
        // driftless diffusion process 
        var_est += sigmaE^2;

      } else {

        winp_mean[i, t] = Pwc0[q];
        winp_var[i, t]  = sigma0[i]^2;
        gain[i, t]      = k;
        rpe[i, t]       = pe;
      }
    }
  }
    
}

