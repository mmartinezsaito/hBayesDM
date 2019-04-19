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

  // hardcoded densities over cues given different models
  Pwc0 = [0, .125, .25, .375, .5, .625, .75, .875, 1]';     
  Pwca = [0, .6, .6, .6, .6, .6, .6, .6, 1]';
}

parameters {
  // group-level parameters
  vector[2]          mu_nc;
  vector<lower=0>[2] sd_nc;

  // sw weighs the contribution of the alternative (actual) model against the default prior model 

   // subject-level raw parameters, follows norm(0,1), for later Matt Trick
  vector[N] Bnu0_nc;    // GAME PRIOR: anticipated initial Haldane Beta prior sample size of the 9 cards
  vector[N] sw_nc;     

}

transformed parameters {
  // subject-level parameters
  vector<lower=0>[N]         Bnu0;         // shape parameter, nu = a + b; mu = a / nu
  vector<lower=0,upper=1>[N] sw;
 

  // Matt Trick
  // the (group) hyperparameters Gaussian location and scale constrain the individual parameters
  for (i in 1:N) {
    Bnu0[i] = exp(        mu_nc[1] + sd_nc[1] * Bnu0_nc[i] );  // there is no upper limit to the (Haldane prior) sample size
    sw[i]  = Phi_approx( mu_nc[2] + sd_nc[2] * sw_nc[i] );
  }

}

model {

  // ======= BAYESIAN PRIORS ======= //
  // group level hyperparameters
  mu_nc ~ normal(0,1);
  sd_nc ~ normal(0,1); // student_t(4,0,1); 
                       //cauchy(0,1);   // why half-Cauchy: Ahn, Haynes, Zhang (2017). 
                       //From v0.6.0, cauchy(0,5) -> cauchy(0,1) 

  // individual parameters: non-centered parameterization
  Bnu0_nc ~ normal(0,1);  // game prior: cognitive flexibility for alternative model search
  sw_nc  ~ std_normal();    
 
 
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
        
        // a priori shape parameters  
        Bnu = Balpha + Bbeta;
        Bmu = Balpha ./ Bnu; 
    
        // weighted parameters  
        Bmw[q] = Bmu[q] * sw[i] + Pwc0[q] * (1 - sw[i]); 
    
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
  real<lower=0, upper=1> mu_sw;
  real<lower=0>          sd_Bnu0;
  real<lower=0>          sd_sw;

  real log_lik[N];

  real beta_mean[N, T];
  real beta_samsiz[N, T];
  real weighted_mean[N, T];

  // For posterior predictive check
  real y_pred[N,T];

  // set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_Bnu0 = exp(       mu_nc[1] );
  mu_sw  = Phi_approx( mu_nc[2] );
  sd_Bnu0 = exp(       sd_nc[1] );
  sd_sw  = Phi_approx( sd_nc[2] );

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

    // ------- GQ ------- //
    log_lik[i] = 0;
    // ------- END GQ ------- //

    for (t in 1:(Tsubj[i])) {
      int q;
      q = cue[i, t];

      if (q != 1 && q != 9) { 

        // a priori shape parameters  
        Bnu = Balpha + Bbeta;
        Bmu = Balpha ./ Bnu; 
    
        // weighted parameters  
        Bmw[q] = Bmu[q] * sw[i] + Pwc0[q] * (1 - sw[i]); 

        // ------- GQ ------- // 
         //log_lik[i] += beta_binomial_lpmf(choice[i, t] | Balpha[q], Bbeta[q]);
         //log_lik[i] += bernoulli_lpmf(choice[i,t] | Pwc0[q]);
        log_lik[i] += bernoulli_lpmf(choice[i,t] | Bmw[q]);

        y_pred[i, t] = bernoulli_rng(Bmw[q]); // generate posterior prediction for current trial

        // Model regressors --> store values before being updated
        beta_mean[i, t] = Bmu[q];
        beta_samsiz[i, t] = Bnu[q];
        weighted_mean[i, t] = Bmw[q];
        // ------- END GQ ------- //

        // update shape parameters   
        if (outcome[i,t] == 1) Balpha[q] += 1;  
        else                   Bbeta[q]  += 1;  

      } else {

        y_pred[i, t] = bernoulli_rng(Pwc0[q]); // generate posterior prediction for current trial
        beta_mean[i, t] = Pwc0[q];
        beta_samsiz[i, t] = Bnu0[i];
        weighted_mean[i, t] = Pwc0[q];
      }
    }
  }

}
