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
  vector<lower=0, upper=1>[9] Pwc0;  // probability of winning given the cue according to the briefing
  vector<lower=0, upper=1>[9] Pwca;  // probability of winning given the actual winning densities

  // hardcoded densities over cues given different models
  Pwc0 = [0, .125, .25, .375, .5, .625, .75, .875, 1]';     
  Pwca = [0, .6, .6, .6, .6, .6, .6, .6, 1]';
}

parameters {
  // group-level parameters
  vector[4]          mu_gp;
  vector<lower=0>[4] sigma_gp;

  // subject-level raw parameters, follows norm(0,1), for later Matt Trick
  vector[N] Bnu0_raw;    // GAME PRIOR: anticipated initial Haldane Beta prior sample size of the 9 cards
  vector[N] swU_raw;     // sw weighs the contribution of the alternative (actual) model against the default prior model 
  vector[N] swK_raw;     // U: upper asymptote, K: growth rate or steepness, 0: sigmoid midpoint
  vector[N] sw0_raw;     

   //real[N] p1_raw;       // probability of winning given the cue if a fixed value, not necessarily 0.6, is assumed  <-------- COMMENT IN/OUT 
}

transformed parameters {
  // subject-level parameters
  vector<lower=0>[N]         Bnu0;         // shape parameter, nu = a + b; mu = a / nu
  vector<lower=0,upper=1>[N] swU;
  vector<lower=0>[N]         swK;
  vector<lower=0,upper=T>[N] sw0;
   //int<lower=0, upper=1> isp1free;
   //vector<lower=0, upper=5/3>[N] p1;    // <----------------  COMMMENT IN/OUT


  // Matt Trick
  // the (group) hyperparameters Gaussian location and scale constrain the individual parameters
  for (i in 1:N) {
    Bnu0[i] = exp(       mu_gp[1] + sigma_gp[1] * Bnu0_raw[i] );  // there is no upper limit to the (Haldane prior) sample size
    swU[i] = Phi_approx( mu_gp[2] + sigma_gp[2] * swU_raw[i] );
    swK[i] = exp(        mu_gp[3] + sigma_gp[3] * swK_raw[i] );
    sw0[i] = Phi_approx( mu_gp[4] + sigma_gp[4] * sw0_raw[i] ) * T;
     // p1[i] = Phi_approx( mu_gp[5] + sigma_gp[5] * p1_raw[i] );    // <------------ COMMENT IN/OUT
  }

}

model {

  // ======= BAYESIAN PRIORS ======= //
  // group level hyperparameters
  mu_gp    ~ normal(0,1);
  sigma_gp ~ cauchy(0,5);   // why half-Cauchy: Ahn, Haynes, Zhang (2017). 
                            //From v0.6.0, cauchy(0,5) -> cauchy(0,1) 

  // individual parameters: non-centered parameterization
  Bnu0_raw ~ normal(0,1);  // game prior: cognitive flexibility for alternative model search
  swU_raw  ~ std_normal();    
  swK_raw  ~ std_normal();    
  sw0_raw  ~ std_normal();    
   //p1          ~ std_normal();    // <---------------- COMMENT IN/OUT


  // ======= LIKELIHOOD FUNCTION ======= //
  // subject loop and trial loop
  for (i in 1:N) {
    vector[9] Bmu;    // estimated Bmu (mean) for each option 
    vector[9] Bnu; // estimated Bnu for each option
    vector[9] Balpha;
    vector[9] Bbeta;
    vector[9] Bmw;

    // initialize shape parameters
    Bmu = Pwc0;                  
    Bnu = rep_vector(Bnu0[i], 9);
    Bmw = Bmu; 

    Balpha = Bmu .* Bnu;
    Bbeta = (1 - Bmu) .* Bnu;

    for (t in 1:(Tsubj[i])) {
      int q;
      q = cue[i, t];

      if (q != 1 && q != 9) { 
        real sw;

        // sw denotes weighing the two models, it's a sigmoid of time 
        sw = swU[i] * logistic_cdf(t , sw0[i], 1 / swK[i]);

        // a priori shape parameters  
        Bnu = Balpha + Bbeta;
        Bmu = Balpha ./ Bnu; 
    
        // weighted parameters  
        Bmw[q] = Bmu[q] * sw + Pwc0[q] * (1 - sw); 
    
        // a priori choice probabilities
         //choice[i, t] ~ beta_binomial(1 | Balpha[q], Bbeta[q]);  
         //choice[i, t] ~ bernoulli(Pwc0[q]);  
        choice[i, t] ~ bernoulli(Bmw[q]);  
    
        // update shape parameters   
        if (outcome[i,t] == 1) Balpha[q] += 1;  
        else                   Bbeta[q]  += 1;  
      }
    }
  }

}

generated quantities {

  real<lower=0>          mu_Bnu0;
  real<lower=0, upper=1> mu_swU;
  real<lower=0>          mu_swK;
  real<lower=0, upper=T> mu_sw0;

  real log_lik[N];

  real beta_mean[N, T];
  real beta_samsiz[N, T];
  real weight_par[N, T];
  real weighted_mean[N, T];

  // For posterior predictive check
  real y_pred[N,T];

  // set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_Bnu0 = exp(mu_gp[1]);
  mu_swU = Phi_approx(mu_gp[2]);
  mu_swK = exp(mu_gp[3]);
  mu_sw0 = Phi_approx(mu_gp[4]) * T;

  // subject and trial loops
  for (i in 1:N) {
    vector[9] Bmu;    // estimated Bmu (mean) for each option 
    vector[9] Bnu; // estimated Bnu for each option
    vector[9] Balpha;
    vector[9] Bbeta;
    vector[9] Bmw;

    // initialize shape parameters
    Bmu = Pwc0;                  
    Bnu = rep_vector(Bnu0[i], 9);

    Balpha = Bmu .* Bnu;
    Bbeta = (1 - Bmu) .* Bnu;

    log_lik[i] = 0;  // ------- GQ ------- 

    for (t in 1:(Tsubj[i])) {
      int q;
      real sw;

      q = cue[i, t];

      // sw denotes weighing the two models, it's a sigmoid of time 
      sw = swU[i] * logistic_cdf(t , sw0[i], 1 / swK[i]);

      if (q != 1 && q != 9) { 

        // a priori shape parameters  
        Bnu = Balpha + Bbeta;
        Bmu = Balpha ./ Bnu; 
    
        // weighted parameters  
        Bmw[q] = Bmu[q] * sw + Pwc0[q] * (1 - sw); 

        // ------- GQ ------- // 
         //log_lik[i] += beta_binomial_lpmf(choice[i, t] | Balpha[q], Bbeta[q]);
         //log_lik[i] += bernoulli_lpmf(choice[i,t] | Pwc0[q]);
        log_lik[i] += bernoulli_lpmf(choice[i,t] | Bmw[q]);

        // generate posterior prediction for current trial
        y_pred[i, t] = bernoulli_rng(Bmw[q]);

        // Model regressors --> store values before being updated
        beta_mean[i, t] = Bmu[q];
        beta_samsiz[i, t] = Bnu[q];
        weight_par[i, t] = sw;
        weighted_mean[i, t] = Bmw[q];
        // ------- END GQ ------- //

        // update shape parameters   
        if (outcome[i,t] == 1) Balpha[q] += 1;  
        else                   Bbeta[q]  += 1;  

      } else {

        beta_mean[i, t] = Pwc0[q];
        beta_samsiz[i, t] = Bnu0[i];
        weight_par[i, t] = sw;
        weighted_mean[i, t] = Pwc0[q];
      }
    }
  }

}
