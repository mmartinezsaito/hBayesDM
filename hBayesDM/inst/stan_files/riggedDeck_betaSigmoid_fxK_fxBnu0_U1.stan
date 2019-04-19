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
  real R;

  // hardcoded densities over cues given different models
  Pwc0 = [0, .125, .25, .375, .5, .625, .75, .875, 1]';     
  Pwca = [0, .6, .6, .6, .6, .6, .6, .6, 1]';

  R = 6; // roll-off of logistic function from center
}

parameters {
  // group-level parameters
  vector[1]          mu_nc;
  vector<lower=0>[1] sd_nc;

  // U: upper asymptote, K: growth rate or steepness, 0: sigmoid midpoint
  // sw weighs the contribution of the alternative (actual) model against the default prior model 

  // Individual fixed effect
  real<lower=0> swK;   
  real<lower=0> Bnu0;   // GAME PRIOR: anticipated initial Haldane Beta prior sample size of the 9 cards   

  // subject-level raw parameters, follows norm(0,1), for later Matt Trick
  vector[N] sw0_nc;    

}

transformed parameters {
  // subject-level parameters
  vector<lower=1,upper=T+R>[N] sw0;
  vector<lower=0,upper=1>[N]   swU;


  // Matt Trick
  // the (group) hyperparameters Gaussian location and scale constrain the individual parameters
  for (i in 1:N) {
    sw0[i]  = Phi_approx( mu_nc[1] + sd_nc[1] * sw0_nc[i] ) * (T-1+R) + 1; // upper limit instead ot *(T-1)+1 to model the outcome U=0
    swU[i] = 1;
  }

}

model {

  // ======= BAYESIAN PRIORS ======= //
  // group level hyperparameters
  mu_nc ~ normal(0,1);
  sd_nc ~ normal(0,1); 
                       // student_t(4,0,1); 
                       //cauchy(0,1);   // why half-Cauchy: Ahn, Haynes, Zhang (2017). 
                       //From v0.6.0, cauchy(0,5) -> cauchy(0,1) 

  // individual fixed effects
  swK  ~ gamma(2, 2);
  Bnu0 ~ gamma(1, .5);

  // individual parameters: non-centered parameterization
  sw0_nc  ~ std_normal();    


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
    Bnu = rep_vector(Bnu0, 9);
    Bmw = Bmu; 

    Balpha = Bmu .* Bnu;
    Bbeta = (1 - Bmu) .* Bnu;

    for (t in 1:(Tsubj[i])) {
      int q;
      q = cue[i, t];

      if (q != 1 && q != 9) { 
        real sw;

        // sw denotes weighing the two models, it's a sigmoid of time 
        sw = swU[i] * logistic_cdf(t , sw0[i], 1 / swK);    // swK=1 corresponds to slope of 1/4 at origin

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

  real<lower=1, upper=T+R> mu_sw0;
  real<lower=0>            sd_sw0;

  real log_lik[N];

  real beta_mean[N, T];
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

  mu_sw0  = Phi_approx( mu_nc[1] ) * (T-1+R) + 1;
  sd_sw0  = Phi_approx( sd_nc[1] ) * (T-1+R) + 1;

  // subject and trial loops
  for (i in 1:N) {
    vector[9] Bmu;    // estimated Bmu (mean) for each option 
    vector[9] Bnu; // estimated Bnu for each option
    vector[9] Balpha;
    vector[9] Bbeta;
    vector[9] Bmw;

    // initialize shape parameters
    Bmu = Pwc0;                  
    Bnu = rep_vector(Bnu0, 9);

    Balpha = Bmu .* Bnu;
    Bbeta = (1 - Bmu) .* Bnu;

    // ------- GQ ------- //
    log_lik[i] = 0;
    // ------- END GQ ------- //

    for (t in 1:(Tsubj[i])) {
      int q;
      real sw;

      q = cue[i, t];

      // sw denotes weighing the two models, it's a sigmoid of time 
      sw = swU[i] * logistic_cdf(t , sw0[i], 1 / swK);

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

        y_pred[i, t] = bernoulli_rng(Bmw[q]); // generate posterior prediction for current trial

        // Model regressors --> store values before being updated
        beta_mean[i, t] = Bmu[q];
        weight_par[i, t] = sw;
        weighted_mean[i, t] = Bmw[q];
        // ------- END GQ ------- //

        // update shape parameters   
        if (outcome[i,t] == 1) Balpha[q] += 1;  
        else                   Bbeta[q]  += 1;  

      } else {

        y_pred[i, t] = bernoulli_rng(Pwc0[q]); // generate posterior prediction for current trial
        beta_mean[i, t] = Pwc0[q];
        weight_par[i, t] = sw;
        weighted_mean[i, t] = Pwc0[q];
      }
    }
  }

}
