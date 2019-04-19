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

  // hardcoded densities over cues given different models
  Pwc0 = [0, .125, .25, .375, .5, .625, .75, .875, 1]';     
}

parameters {
}

transformed parameters {
}

model {

  // ======= LIKELIHOOD FUNCTION ======= //
  // subject loop and trial loop
  for (i in 1:N) {

    for (t in 1:(Tsubj[i])) {
      int q;
      q = cue[i, t];

      if (q != 1 && q != 9) { 
        choice[i, t] ~ bernoulli(Pwc0[q]);  
      }
    }
  }

}

generated quantities {

  real log_lik[N];

  // For posterior predictive check
  real y_pred[N,T];

  // set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  // subject and trial loops
  for (i in 1:N) {

    log_lik[i] = 0;  // ------- GQ ------- 

    for (t in 1:(Tsubj[i])) {
      int q;
      q = cue[i, t];

      // generate posterior prediction for current trial
      y_pred[i, t] = bernoulli_rng(Pwc0[q]);

      if (q != 1 && q != 9) { 
        // ------- GQ ------- // 
        log_lik[i] += bernoulli_lpmf(choice[i,t] | Pwc0[q]);
        // ------- END GQ ------- //
      }
    }
  }

}
